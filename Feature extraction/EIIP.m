function A = N2EIIP( data )
%二核苷酸的电子-离子相互作用赝势
%%计算相互作用系数
ea=0.126;
et=0.1335;
eg=0.0806;
ec=0.134;
eACGT=[ea ec eg et];
t=1;
emer=zeros(1,16);
for i=1:4
    first=eACGT(1,i);
    for j=1:4
        second=eACGT(1,j);
        emer(1,t)=first*second;%%+/*
        t=t+1;
    end
end


%%计算2mer
AA='ACGT';
m=length(AA);%4
r=0;
for i=1:m
       x1=AA(i);
    for j=1:m
         x2=AA(j);
           x=strcat(x1,x2);
           r=r+1;
           H{1,r}=x;           
    end  
end
len=length(data);
N2=zeros(len,m^2);
data=data';
for i=1:len
    s=data{1,i};
    M=length(s);
    for j=1:M-1
    a1=s(j);
    a2=s(j+1);
    a=strcat(a1,a2);
    g=strmatch(a,H,'exact');
    N2(i,g)=N2(i,g)+1;
    end
    N2(i,:)=N2(i,:)/(M-1);
end

emer=ones(size(data,1),1)*emer;
A=N2.*emer;
end