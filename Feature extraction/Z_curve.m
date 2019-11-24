function xyz=Z_curve(data)
%%计算2mer的Z曲线
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

%%计算xyz
xyz=zeros(len,12);
for i=1:len
    x=zeros(4,1);
    y=zeros(4,1);
    z=zeros(4,1);
    for j=1:4
        x(j)=(N2(i,4*(j-1)+1)+N2(i,4*(j-1)+3))-(N2(i,4*(j-1)+2)-N2(i,4*(j-1)+4));
        y(j)=(N2(i,4*(j-1)+1)+N2(i,4*(j-1)+2))-(N2(i,4*(j-1)+3)-N2(i,4*(j-1)+4));
        z(j)=(N2(i,4*(j-1)+1)+N2(i,4*(j-1)+4))-(N2(i,4*(j-1)+2)-N2(i,4*(j-1)+3));
    end
    xyz(i,:)=reshape([x y z],[1,12]);
end
