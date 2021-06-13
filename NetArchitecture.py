import torch
# Definir arquitectura de la red; para ello derivamos
# de torch.nn.Module
# Podemos configurar todas las capas que queramos
class NetArchitecture(torch.nn.Module):
    def __init__(self):
        super(NetArchitecture, self).__init__()
        self.layer01 = torch.nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.layer02 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer03 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.layer04 = torch.nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.layer05 = torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.layer06 = torch.nn.Conv2d(160, 32, kernel_size=3, stride=1, padding=1)
        self.layer07 = torch.nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1)
        self.layer08 = torch.nn.Conv2d(224, 32, kernel_size=3, stride=1, padding=1)
        self.layerout = torch.nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        


    # Una vez definida la arquitectura, lo único que me pide Torch
    # es la implementación forward, es decir, como voy a pasar los datos
    # a estos módulos.
    def forward(self, x):
        x1 = torch.relu(self.layer01(x))
        x2 = torch.relu(self.layer02(x1))
        c2_dense = torch.relu(torch.cat([x1, x2], 1))
        
        x3 = torch.relu(self.layer03(c2_dense))
        c3_dense = torch.relu(torch.cat([x1, x2, x3], 1))
        
        x4 = torch.relu(self.layer04(c3_dense))
        c4_dense = torch.relu(torch.cat([x1, x2, x3, x4], 1))
        
        x5 = torch.relu(self.layer05(c4_dense))
        c5_dense = torch.relu(torch.cat([x1, x2, x3, x4, x5], 1))
        
        x6 = torch.relu(self.layer06(c5_dense))
        c6_dense = torch.relu(torch.cat([x1, x2, x3, x4, x5, x6], 1))
        
        x7 = torch.relu(self.layer07(c6_dense))
        c7_dense = torch.relu(torch.cat([x1, x2, x3, x4, x5, x6,x7], 1))
        
        x8 = torch.relu(self.layer08(c7_dense))
        c8_dense = torch.relu(torch.cat([x1, x2, x3, x4, x5, x6,x7,x8], 1))
        
        x9 = torch.relu(self.layerout(c8_dense))
        
        return x9


#1st architecture
        '''
        x = torch.relu(self.layer01(x))
        x = torch.relu(self.layer02(x))
        x = torch.relu(self.layer03(x))
        x = torch.relu(self.layer04(x))
        x = torch.relu(self.layer05(x))
        x = torch.relu(self.layer06(x))
        x = torch.relu(self.layerout(x))
        '''

# 2nd architecture
'''
x = torch.relu(self.layer01(x))
        y = x

        x = torch.relu(self.layer02(x))
        x = torch.relu(self.layer03(x))
        z = x
        x = x + y

        x = torch.relu(self.layer04(x))
        x = torch.relu(self.layer05(x))
        y = x
        x = x + z

        x = torch.relu(self.layer06(x))
        x = torch.relu(self.layerout(x))
'''

'''
self.bn = torch.nn.BatchNorm2d(6)
self.conv1 = torch.nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
self.conv3 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
self.conv4 = torch.nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
self.conv5 = torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
self.conv6 = torch.nn.Conv2d(160, 32, kernel_size=3, stride=1, padding=1)
self.layerout = torch.nn.Conv2d(192, 3, kernel_size=3, stride=1, padding=1)


conv1 = torch.relu(self.conv1(self.bn(x)))
conv2 = torch.relu(self.conv2(conv1))
c2_dense = torch.relu(torch.cat([conv1, conv2], 1))

conv3 = torch.relu(self.conv3(c2_dense))
c3_dense = torch.relu(torch.cat([conv1, conv2, conv3], 1))
        
conv4 = torch.relu(self.conv4(c3_dense)) 
c4_dense = torch.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
        
conv5 = torch.relu(self.conv5(c4_dense))
c5_dense = torch.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

conv6 = torch.relu(self.conv6(c5_dense))
c6_dense = torch.relu(torch.cat([conv1, conv2, conv3, conv4, conv5,conv6], 1))

conv7 = torch.relu(self.layerout(c6_dense))
c7_dense = torch.relu(torch.cat([conv1, conv2, conv3, conv4, conv5,conv6,conv7], 1))

'''