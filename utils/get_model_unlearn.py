import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from torchvision import models
from .pytorchtools import EarlyStopping

def top_k_accuracy(outputs, labels, k=5):
    _, top_k_predictions = torch.topk(outputs, k, dim=1)  # Get the indices of the top k predictions
    correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))  # Check correctness
    top_k_acc = correct.sum().float() / labels.size(0)  # Calculate accuracy
    return top_k_acc

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv3x3(inplanes, width, stride)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, expand=1.0, block=BasicBlock, layers=[1, 1, 1, 1],  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(16 * expand)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * expand), layers[0])
        self.layer2 = self._make_layer(block, int(128 * expand), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * expand), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * expand), layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(int(512 * block.expansion * expand), num_classes[0])
        self.fc_2 = nn.Linear(int(512 * block.expansion * expand), num_classes[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_1 = self.fc_1(x)
        x_2 = self.fc_2(x)

        return x_1, x_2

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, expand=1.0):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(16 * expand)
        layers = [conv_bn(in_channels, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * expand)
            for i in range(n):
                if i == 0:
                    layers.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                else:
                    layers.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        output_channel = int(1280 * expand) if expand > 1.0 else 1280
        layers.append(conv_1x1_bn(input_channel, output_channel))
        self.features = nn.Sequential(*layers)
        # building classifier
        self.classifier_1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes[0]),
        )

        self.classifier_2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes[1]),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x_1 = self.classifier_1(x)
        x_2 = self.classifier_2(x)
        return x_1, x_2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class TransformerModel(nn.Module):
    def __init__(self, in_channels, num_classes, expand=1.0, depth=12):
        super(TransformerModel, self).__init__()        
        width = int(64*expand)
        self.transformer = models.VisionTransformer(image_size=48, patch_size=4, num_layers=depth, \
                                hidden_dim=width, mlp_dim=width, num_heads=8, num_classes=num_classes[0])
        self.transformer.conv_proj = nn.Conv2d(in_channels, width, kernel_size=(4, 4), stride=(4, 4))
        self.transformer.heads.head = nn.Identity()

        self.head1 = nn.Linear(width, num_classes[0])
        self.head2 = nn.Linear(width, num_classes[1])

    def forward(self, x):
        x = self.transformer(x)
        x_1 = self.head1(x)
        x_2 = self.head2(x)
        return x_1, x_2

# Define the simple architecture
class SimpleModel(nn.Module):
    def __init__(self, in_channels, num_classes, expand=1.0, idx=14): # expland: []
        super(SimpleModel, self).__init__()

        fv = [144, 288, 432, 576, 576, 720, 864, 1008, 1152, 1152, 1296, 1440, 1584, 1728, 2304]

        base_ch = int(16*expand)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_ch, base_ch*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_ch*2, base_ch*2*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

        )
        self.classifier_1 = nn.Sequential(nn.Linear(fv[idx], base_ch*2),
                                          nn.ReLU(),
                                          nn.Linear(base_ch*2, num_classes[0]))
        
        self.classifier_2 = nn.Sequential(nn.Linear(fv[idx], base_ch*2),
                                          nn.ReLU(),
                                          nn.Linear(base_ch*2, num_classes[1]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x_1 = self.classifier_1(x)
        x_2 = self.classifier_2(x)
        return x_1, x_2

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model_er(model, beta, num_epochs, optimizer, criterion, train_loader, val_loader, test_loader, len_train, len_val, len_test, device):

    early_stopping = EarlyStopping(patience=5, verbose=False)
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, idents, classes in train_loader:
            images = images.to(device)
            idents = idents.to(device).long()
            classes = classes.to(device).long()

            optimizer.zero_grad()

            outputs, _ = model(images)

            loss = criterion(outputs, classes)

            loss.backward()

            _, class_1_outputs = model(images)

            class_1_loss = (-beta) * criterion(class_1_outputs, idents)
            class_1_loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len_train

        running_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        running_corrects = 0
        
        for images, idents, classes in val_loader:
            images = images.to(device)
            idents = idents.to(device).long()
            classes = classes.to(device).long()
            
            outputs, _ = model(images)
            loss = criterion(outputs, classes)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == classes).item()


            running_loss += loss.item() * images.size(0)

        val_loss = running_loss / len_val

        accuracy = running_corrects / len_val
        
        if accuracy > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = accuracy

        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            break
    
    # Evaluation loop
    best_model.eval()  # Set the model to evaluation mode
    running_corrects = 0

    with torch.no_grad():
        for images, idents, classes in test_loader:
            images = images.to(device)
            idents = idents.to(device).long()
            classes = classes.to(device).long()

            outputs, _ = best_model(images)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == classes).item()


    accuracy = running_corrects / len_test
            
    return best_model, accuracy, val_loss

def train_model_pneu(model, beta, num_epochs, optimizer, criterion, train_loader, val_loader, test_loader, len_train, len_val, len_test, device):

    early_stopping = EarlyStopping(patience=5, verbose=False)
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, classes, pathologies in train_loader:
            images = images.to(device)
            pathologies = pathologies.to(device).long()
            classes = classes.to(device).long()

            optimizer.zero_grad()

            outputs, _ = model(images)

            loss = criterion(outputs, classes)

            loss.backward()

            class_1_inputs = []
            class_1_labels = []

            for images, classes, pathology, in zip(images, pathologies, classes):
                if pathology.item() != 0:
                    class_1_inputs.append(images)
                    class_1_labels.append(pathology)

            if class_1_inputs:  # Check if there are any samples satisfying the condition
                class_1_inputs = torch.stack(class_1_inputs , dim=0)  # Stack the list of tensors to form a single tensor
                class_1_labels = torch.stack(class_1_labels, dim=0).long()  # Stack the list of tensors to form a single tensor
                
                _, class_1_outputs = model(class_1_inputs)

                class_1_loss = (-beta) * criterion(class_1_outputs, class_1_labels)

                class_1_loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len_train

        running_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        running_corrects = 0
        
        for images, classes, pathologies in val_loader:
            images = images.to(device)
            pathologies = pathologies.to(device).long()
            classes = classes.to(device).long()
            
            outputs, _ = model(images)
            loss = criterion(outputs, classes)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == classes).item()


            running_loss += loss.item() * images.size(0)

        val_loss = running_loss / len_val

        accuracy = running_corrects / len_val
        
        if accuracy > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = accuracy

        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            break
    
    # Evaluation loop
    best_model.eval()  # Set the model to evaluation mode
    running_corrects = 0

    with torch.no_grad():
        for images, classes, pathologies in test_loader:
            images = images.to(device)
            pathologies = pathologies.to(device).long()
            classes = classes.to(device).long()

            outputs, _ = best_model(images)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == classes).item()


    accuracy = running_corrects / len_test
            
    return best_model, accuracy, val_loss



class SubModelN(nn.Module):
    def __init__(self, model, n_layers, model_name, setting):
        super(SubModelN, self).__init__()

        self.model_name = model_name
        self.setting = setting
        grad_sample_module = model._module if isinstance(model, GradSampleModule) else model

        for i, layer in enumerate(grad_sample_module.children()):
            
            if i < n_layers:
                self.add_module(f'layer{i}', layer)
            
    def forward(self, x):
        for i, layer in enumerate(self.children(), start=0):
            if self.model_name == 'simple' and i == len(list(self.children()))-1:
                x = x.view(x.size(0), -1)

            if self.setting == 'white' and self.model_name == 'resnet':
                x = layer(x)

            if self.model_name == 'resnet' and i == len(list(self.children()))-1:
                x = torch.flatten(x, 1)

            if (self.model_name != 'resnet' or self.setting == 'black'):
                x = layer(x)
        
            if self.model_name == 'mobilenet' and i == 0:
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)

        return x
    

def GetSubModelFromConv(model, conv_idx=2):
                
        desired_submodel = nn.Sequential()
        for name, layer in model.named_children():
            desired_submodel.add_module(name, layer)
            if name == 'conv'+str(conv_idx):
                break

        return desired_submodel