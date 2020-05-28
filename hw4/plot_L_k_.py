import numpy as np
import matplotlib.pyplot as plt  

epoch = np.array([i*10 +1 for i in range(10)])
L_k_train_vae = [-171.8969,-146.4946,-139.5801,-139.11766,-137.65387,-141.1931,-142.0981,-135.63329,-133.35852,-140.50623]
L_k_test_vae = [-168.59242,-146.59343,-142.25009,-148.05525,-146.42516,-142.7102,-141.54662,-144.64365,-144.6463,-134.282]
wake_loss = [219.190562744,198.732230169,177.369796614,169.883391807,168.201636131,169.951192322,171.359023160,170.747817799,170.284503340,169.911900385]
sleep_loss = [13.279088593,3.169254938,1.099091359,-0.152513210,-0.393533209,-0.310044148,-0.233888664,-0.139054484,-0.105226825,0.037620364]
"""
plt.plot(epoch,L_k_test_vae)
plt.title(r'$L^{test}_{100}$ vs. the epoch number of AVEB')
plt.xlabel('epoch')
plt.ylabel(r'$L_k$')
plt.show()
plt.plot(epoch,L_k_train_vae)
plt.title(r'$L^{train}_{100}$ vs. the epoch number of AVEB')
plt.xlabel('epoch')
plt.ylabel(r'$L_k$')
plt.show()
"""

plt.plot(epoch,wake_loss)
plt.title('training losses for the wake-phase vs. the epoch number')
plt.xlabel('epoch')
plt.ylabel('wake loss')
plt.show()
plt.plot(epoch,sleep_loss)
plt.title('training losses for the sleep-phase  vs. the epoch number')
plt.xlabel('epoch')
plt.ylabel('sleep loss')
plt.show()

'''

Epoch: 0001 Wake cost= 219.190562744 Sleep cost= 13.279088593
Epoch: 0011 Wake cost= 198.732230169 Sleep cost= 3.169254938
Epoch: 0021 Wake cost= 177.369796614 Sleep cost= 1.099091359
Epoch: 0031 Wake cost= 169.883391807 Sleep cost= -0.152513210
Epoch: 0041 Wake cost= 168.201636131 Sleep cost= -0.393533209
Epoch: 0051 Wake cost= 169.951192322 Sleep cost= -0.310044148
Epoch: 0061 Wake cost= 171.359023160 Sleep cost= -0.233888664
Epoch: 0071 Wake cost= 170.747817799 Sleep cost= -0.139054484
Epoch: 0081 Wake cost= 170.284503340 Sleep cost= -0.105226825
Epoch: 0091 Wake cost= 169.911900385 Sleep cost= 0.037620364
'''
"""
vae

train
Epoch: 0001 Train cost= 197.830829468
L_k= -171.8969
Epoch: 0011 Train cost= 154.512450395
L_k= -146.4946
Epoch: 0021 Train cost= 149.693050371
L_k= -139.5801
Epoch: 0031 Train cost= 147.187346774
L_k= -139.11766
Epoch: 0041 Train cost= 145.471423423
L_k= -137.65387
Epoch: 0051 Train cost= 144.572841991
L_k= -141.1931
Epoch: 0061 Train cost= 143.796372320
L_k= -142.0981
Epoch: 0071 Train cost= 142.879860424
L_k= -135.63329
Epoch: 0081 Train cost= 142.453738389
L_k= -133.35852
Epoch: 0091 Train cost= 142.006537600
L_k= -140.50623


test
Epoch: 0001 Train cost= 195.648152854
L_k= -168.59242
Epoch: 0011 Train cost= 154.552147883
L_k= -146.59343
Epoch: 0021 Train cost= 150.756386219
L_k= -142.25009
Epoch: 0031 Train cost= 148.966518139
L_k= -148.05525
Epoch: 0041 Train cost= 147.370912198
L_k= -146.42516
Epoch: 0051 Train cost= 146.241383861
L_k= -142.7102
Epoch: 0061 Train cost= 145.345530063
L_k= -141.54662
Epoch: 0071 Train cost= 144.368218786
L_k= -144.64365
Epoch: 0081 Train cost= 143.698566367
L_k= -144.6463
Epoch: 0091 Train cost= 143.222016269
L_k= -134.282
"""