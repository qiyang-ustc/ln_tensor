import torch

def init_ln_tensor(data,sign):
    temp = torch.ones_like(data,dtype=torch.float64)
    temp = ln_tensor(temp)
    temp.data = data
    temp.sign = sign
    return temp

def lnsumexp_sign(data,sign):
    max_data = data.max()
    data = data - max_data
    data = torch.exp(data)
    data = torch.sum(sign*data)
    temp = torch.log(torch.abs(data))
    sign = torch.sign(data)
    #print(temp)
    if torch.isnan(temp):
        #print('nan appear!!')
        return torch.tensor([[float('-inf')]],dtype=torch.float64),sign
    return (max_data+temp),sign

class ln_tensor(object):
    def __init__(self,A):
        if isinstance(A,ln_tensor):
            self.data = A.data
            self.sign = A.sign
        else:
            if not isinstance(A,torch.DoubleTensor):
                raise("ln_tensor should be a DoubleTensor!")
            self.data = torch.log(torch.abs(A))
            self.sign = torch.sign(A)

    def t(self):
        temp = init_ln_tensor(self.data,self.sign)
        temp.data = temp.data.t()
        temp.sign = temp.sign.t()
        return temp
    
    def mm(self,B):
        if not isinstance(B,ln_tensor):
            raise("Another tensor should be ln_tensor")
        if len(self.shape)!=2 or len(B.shape)!=2:
            raise("This tensor and Another tensor should be matrix!")
        if self.shape[1] != B.shape[0]:
            raise Exception("Dimension Error in Matrix Production!")
        
        dim0 = self.shape[0]
        dim1 = B.shape[1]
        dimn = self.shape[1]

        C = ln_tensor(torch.randn((dim0,dim1),dtype=torch.float64))
        C.data = torch.zeros((dim0,dim1),dtype=torch.float64)
        C.sign = torch.zeros((dim0,dim1),dtype=torch.float64)
        temp = torch.empty(dimn,dtype = torch.float64)
        sign_temp = torch.empty(dimn,dtype=torch.float64)
        for i in range(dim0):
            for j in range(dim1):
                temp = self.data[i,:]+B.data[:,j]
                sign_temp = self.sign[i,:]*B.sign[:,j]
                t1,t2 = lnsumexp_sign(temp,sign_temp)
                C.data[i,j] = (C.data[i,j]+t1).clone()
                C.sign[i,j] = (C.sign[i,j]+t2).clone()
        return C

    def to_tensor(self):
        temp = torch.exp(self.data)
        temp = self.sign*temp
        return temp

    @property
    def shape(self):
        return self.data.shape

    def permute(self,*dims):
        temp = init_ln_tensor(self.data,self.sign)
        temp.data = temp.data.permute(*dims)
        temp.sign = temp.sign.permute(*dims)
        return temp
    
    def contiguous(self):
        temp = init_ln_tensor(self.data,self.sign)
        temp.data = temp.data.contiguous()
        temp.sign = temp.sign.contiguous()
        return temp
    
    def view(self,*dims):
        temp = init_ln_tensor(self.data,self.sign)
        temp.data = temp.data.view(*dims)
        temp.sign = temp.sign.view(*dims)
        return temp


if __name__ == '__main__':
    a = torch.randn((2,2,2,2),dtype=torch.float64)
    print(a.contiguous().permute(0,1,3,2) - ln_tensor(a).contiguous().permute(0,1,3,2).to_tensor())
    a = torch.ones((2,2,2),dtype=torch.float64)
    a[0,0,0]=1
    a[0,1,0]=2
    a[1,0,0]=3
    a[1,1,0]=-4
    a[0,0,1]=5
    a[0,1,1]=-2
    a[1,0,1]=-7
    a[1,1,1]=8
    a.exp_()
    b = ln_tensor(a*torch.randn_like(a))
    b = b.contiguous().view(2,4)
    c = ln_tensor(a).contiguous().view(2,4).t()
    d = init_ln_tensor(b.data,b.sign)
    print(c.data,d.data)
    print('\n')
    m = c.mm(d)
    # print(m.data,m.sign)
    # print(torch.log(abs(c.to_tensor().mm(d.to_tensor()))))
    # print(torch.sign(c.to_tensor().mm(d.to_tensor())))
    print('difference=',(c.to_tensor().mm(d.to_tensor())-m.to_tensor()).norm().item())