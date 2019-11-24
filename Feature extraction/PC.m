function feature=PhysChem2(data)
%%双核苷酸理化性质*2mer
Pg=xlsread('PC12_2.xlsx');%理化性质表格
%Pg=xlsread('PC6_2.xlsx')
len=length(data);
feature=zeros(len,192);
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

N2=zeros(len,16);
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

for i=1:len
    once=zeros(16,12);
    for j=1:16
        for k=1:12
            once(j,k)=Pg(j,k)*N2(i,j);
        end
    end
    feature(i,:)=reshape(once,[1,192]);
end
end

        