# --xor

模型使用只有1个隐层的全连接，输入层2个节点，隐层20个，输出层8个，使用MSELoss
数据为由~7的8个数字构成64个数字对，输入数据是一个1乘2的张量，在第一个隐层做hardtanh(wx + b), 在输出层做另一个hardtanh(wx + b)，标签为8个非0即1的数组，数组的index为异或数字对的目标值。
目前正确率只有40%，但是模型应该收敛了，为什么啊！气
