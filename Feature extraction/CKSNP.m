function CKSNP=CKSNP(data,G)
%%DNA、针对序列长度不同的序列g-gap碱基对组成
%变量 G(空格数)
%N=G+1
AA='ACGT';
m=length(AA);%4
r=0;
len=size(data,1);
data=data';
for i=1:m
       x1=AA(i);
    for j=1:m
         x2=AA(j);
           x=strcat(x1,x2);
           r=r+1;
           H{1,r}=x;           
    end  
end
CKSNP=zeros(len,m^2*(G+1));
for ii=1:len
    s=data{1,ii};%样本
    L=length(s);   %此样本长度
    for g=0:G
         PPT1=zeros(1,m^2);%每次重置矩阵（向量）
      for jj=1:L-g-1
       a1=s(jj);
       a2=s(jj+g+1);
       a=strcat(a1,a2);
       t=strmatch(a,H,'exact');
       PPT1(1,t)=PPT1(1,t)+1;
      end
       CKSNP(ii,(16*g+1):16*(g+1))=PPT1(1,:)/(L-g-1);
    end
end
