import torch
import torch.nn
import torch.optim
from torchvision import models, datasets, transforms

train_dataset = datasets.CIFAR10("./", train = True, transform = transforms.ToTensor(), download = True)
test_dataset = datasets.CIFAR10("./", train = False, transform = transforms.ToTensor(), download = True)

BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
device = torch.device("cuda")

## teacher model training
## BaseLine training
teacher = models.resnet50(pretrained=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teacher.parameters(), lr = 0.0002)
teacher.fc = torch.nn.Linear(2048, 10)
teacher = teacher.to(device)

def normal_train(epoch, model, criterion, optimizer, dataloader):
    model.train()
    
    total_loss = 0.0
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        
        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        
    print( f'{epoch+1} training.. -> {epoch+1} total loss result : {total_loss / idx} ')

for epoch in range(30):
    normal_train(epoch, teacher, criterion, optimizer, train_loader)



    ## Teacher model validation
## Teacher model baseline
def normal_eval(model, dataloader):
    model.eval()
    
    correct = 0
    total = 0
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
    
        with torch.no_grad():
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print( f'accuracy : {100 * correct / total} ')
    
normal_eval(teacher, test_loader)


## Student model training
## BaseLine training
student = models.resnet50(pretrained=False) #No pretrained
student.fc = torch.nn.Linear(2048, 10)
student = student.to(device)

criterion_s = torch.nn.CrossEntropyLoss()
optimizer_s = torch.optim.Adam(student.parameters(), lr = 0.0002)


def normal_train(epoch, model, criterion, optimizer, dataloader):
    model.train()
    
    total_loss = 0.0
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        
        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        
    print( f'{epoch+1} training.. -> {epoch+1} total loss result : {total_loss / idx} ')
    
for epoch in range(30):
    normal_train(epoch, student, criterion_s, optimizer_s, train_loader)




#KD training
#teacher : Resnet50(True) / Student : Resnet18(False)
import torch.nn.functional as F

student_kd = models.resnet50(pretrained=False) #No pretrained
student_kd.fc = torch.nn.Linear(2048, 10)
student_kd = student_kd.to(device)

optimizer = torch.optim.Adam(student_kd.parameters(), lr = 0.0002)

def loss_fn_kd(student_output, teacher_output, labels):
    #Key point is probability kl_div : calculated information distance between Student and Teacher.
    alpha = 0.9 #The Paper explains to upgrade KD method to use higer proportion of A 
    T = 20 #Normal Temperature -> Reason of distillation / high Temperature value make a very soft prediction or labels
    soft_label = F.softmax(teacher_output/T, dim = 1)
    soft_prediction = F.log_softmax(student_output/T, dim = 1)
    dist_loss = torch.nn.functional.kl_div(soft_prediction, soft_label, reduction ="batchmean")
    kd_loss = ( dist_loss * alpha * T * T) + (F.cross_entropy(student_output, labels) * (1 - alpha))  # L(KD) + L(CE) / 업데이트 하면서 이득을 주려고 하는 것.
    
    return kd_loss
    

def kd_train(epoch, t_model, s_model, optimizer, dataloader_train):
    t_model.eval()
    s_model.eval()
    total_loss = 0.0
    
    for idx, (images, labels) in enumerate(dataloader_train):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            t_output = t_model(images) #distillation
        s_output = s_model(images) #distilled
        
        optimizer.zero_grad()
        loss = loss_fn_kd(s_output, t_output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        
    print( f'{epoch+1} training.. -> {epoch+1} total loss result : {total_loss / idx} ')

for epoch in range(30):
    kd_train(epoch, teacher,student_kd, optimizer, train_loader)


#Distillation Successful !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
normal_eval(student_kd, test_loader)
