from gatgnn.data                   import *
from gatgnn.model                  import *
from gatgnn.paddle_early_stopping  import *
from gatgnn.file_setter            import use_property
from gatgnn.utils                  import *

# MOST CRUCIAL DATA PARAMETERS
parser = argparse.ArgumentParser(description='GATGNN')
parser.add_argument('--property', default='bulk-modulus',
                    choices=['absolute-energy','band-gap','bulk-modulus',
                             'fermi-energy','formation-energy',
                             'poisson-ratio','shear-modulus','new-property'],
                    help='material property to train (default: bulk-modulus)')
parser.add_argument('--data_src', default='CGCNN',choices=['CGCNN','MEGNET','NEW'],
                    help='selection of the materials dataset to use (default: CGCNN)')

# MOST CRUCIAL MODEL PARAMETERS
parser.add_argument('--num_layers',default=3, type=int,
                    help='number of AGAT layers to use in model (default:3)')
parser.add_argument('--num_neurons',default=64, type=int,
                    help='number of neurons to use per AGAT Layer(default:64)')
parser.add_argument('--num_heads',default=4, type=int,
                    help='number of Attention-Heads to use  per AGAT Layer (default:4)')
parser.add_argument('--use_hidden_layers',default=True, type=bool,
                    help='option to use hidden layers following global feature summation (default:True)')
parser.add_argument('--global_attention',default='composition', choices=['composition','cluster']
                    ,help='selection of the unpooling method as referenced in paper GI M-1 to GI M-4 (default:composition)')
parser.add_argument('--cluster_option',default='fixed', choices=['fixed','random','learnable'],
                    help='selection of the cluster unpooling strategy referenced in paper GI M-1 to GI M-4 (default: fixed)')
parser.add_argument('--concat_comp',default=False, type=bool,
                    help='option to re-use vector of elemental composition after global summation of crystal feature.(default: False)')
parser.add_argument('--train_size',default=0.8, type=float,
                    help='ratio size of the training-set (default:0.8)')
args = parser.parse_args(sys.argv[1:])

# GATGNN --- parameters
crystal_property                      = args.property
data_src                              = args.data_src
source_comparison, training_num, RSM  = use_property(crystal_property,data_src)
norm_action, classification           = set_model_properties(crystal_property)
if training_num == None: training_num = args.train_size

number_layers                        = args.num_layers
number_neurons                       = args.num_neurons
n_heads                              = args.num_heads
xtra_l                               = args.use_hidden_layers 
global_att                           = args.global_attention
attention_technique                  = args.cluster_option
concat_comp                          = args.concat_comp

# PRINTING PARAMETERS
print(f'> PROPERTY: {crystal_property} | DATA-SOURCE: {data_src}')
print(f'> MODEL: {number_layers} LAYERS | {number_neurons} NEURONS | {n_heads} HEADS')
print(f'> GLOBAL-ATTENTION: {global_att} | CLUSTER-OPTION: {attention_technique} | CONCAT-COMP: {concat_comp}')
print(f'> TRAINING-SIZE: {training_num} | SOURCE-COMPARISON: {source_comparison} | RSM: {RSM}')
print(f'> NORMALIZATION: {norm_action} | CLASSIFICATION: {classification}')

# SETTING UP CODE TO RUN ON GPU
gpu_id = 3
paddle.device.set_device(f"gpu:{gpu_id}")

# DATA PARAMETERS
random_num          =  456
random.seed(random_num)

# MODEL HYPER-PARAMETERS
num_epochs      = 200
learning_rate   = 5e-3
batch_size      = 256

stop_patience   = 150
best_epoch      = 1
adj_epochs      = 50
milestones      = [150,250]
train_param     = {'batch_size':batch_size, 'shuffle': True}
valid_param     = {'batch_size':256, 'shuffle': True}

# DATALOADER/ TARGET NORMALIZATION
src_CIF         = 'CIF-DATA_NEW' if data_src == 'NEW' else 'CIF-DATA'
dataset         = pd.read_csv(f'DATA/{src_CIF}/id_prop.csv',names=['material_ids','label']).sample(frac=1,random_state=random_num)
NORMALIZER      = DATA_normalizer(dataset.label.values)

CRYSTAL_DATA    = CIF_Dataset(dataset, root_dir = f'DATA/{src_CIF}/',**RSM)
idx_list        = list(range(len(dataset)))
random.shuffle(idx_list)

train_idx,test_val = train_test_split(idx_list,train_size=training_num,random_state=random_num)
_,       val_idx   = train_test_split(test_val,test_size=0.5,random_state=random_num)

training_set       =  CIF_Lister(train_idx,CRYSTAL_DATA,NORMALIZER,norm_action,df=dataset,src=data_src)
validation_set     =  CIF_Lister(val_idx,CRYSTAL_DATA,NORMALIZER,norm_action,  df=dataset,src=data_src)

# NEURAL-NETWORK
net = GATGNN(n_heads,classification,neurons=number_neurons,nl=number_layers,xtra_layers=xtra_l,global_attention=global_att,
                unpooling_technique=attention_technique,concat_comp=concat_comp,edge_format=data_src)

# LOSS & OPTMIZER & SCHEDULER
if classification == 1: criterion   = nn.CrossEntropyLoss(); funct = paddle_accuracy
else                  : criterion   = nn.SmoothL1Loss()    ; funct = paddle_MAE
scheduler         = lr.MultiStepDecay(learning_rate, milestones=milestones, gamma=0.3)
optimizer         = optim.AdamW(learning_rate = scheduler, parameters=net.parameters(), weight_decay = 1e-1)


# EARLY-STOPPING INITIALIZATION
early_stopping = EarlyStopping(patience=stop_patience, increment=1e-6,verbose=True,save_best=True,classification=classification)

# METRICS-OBJECT INITIALIZATION
metrics        = METRICS(crystal_property,num_epochs,criterion,funct)

print(f'> TRAINING MODEL ...')
train_loader   = paddle_loader(dataset=training_set,   **train_param)
valid_loader   = paddle_loader(dataset=validation_set, **valid_param) 

for epoch in range(num_epochs):
    # TRAINING-STAGE
    net.train()       
    start_time       = time.time()
    for data in train_loader:
        data         = data
        predictions  = net(data)
        train_label  = metrics.set_label('training',data)
        loss         = metrics('training',predictions,train_label,1)
        _            = metrics('training',predictions,train_label,2)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        metrics.training_counter+=1
    metrics.reset_parameters('training',epoch)

    # VALIDATION-PHASE
    net.eval()
    for data in valid_loader:
        data = data
        predictions    = net(data)
        valid_label        = metrics.set_label('validation',data)
        _                  = metrics('validation',predictions,valid_label,1)
        _                  = metrics('validation',predictions, valid_label,2)

        metrics.valid_counter+=1
    metrics.reset_parameters('validation',epoch)

    scheduler.step()
    end_time         = time.time()
    e_time           = end_time-start_time
    metrics.save_time(e_time)
    
    # EARLY-STOPPING
    early_stopping(metrics.valid_loss2[epoch], net)
    flag_value = early_stopping.flag_value+'_'*(22-len(early_stopping.flag_value))
    if early_stopping.FLAG == True:    estop_val = flag_value
    else:
        estop_val        = '@best: saving model...'; best_epoch = epoch+1
    output_training(metrics,epoch,estop_val,f'{e_time:.1f} sec.')

    if early_stopping.early_stop:
        print("> Early stopping")
        break

# SAVING MODEL
print(f"> DONE TRAINING !")
shutil.copy2('TRAINED/crystal-checkpoint.pdparams', f'TRAINED/{crystal_property}.pdparams')
