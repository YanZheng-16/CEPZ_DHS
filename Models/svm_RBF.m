clear
% result=[ bestSn,bestSp,bestacc,bestMCC ];
%% ������
load('C.mat')
data=X;
% ��zscore��׼��֮�������
data=zscore(data);
[bestc, bestg, bestacc, bestSn, bestSp, bestMCC] = svmcg(data,1,-5,10,1,-5,10, 0,0,0);
%% svmcg����
function [bestc, bestg, bestacc, bestSn, bestSp, bestMCC] = svmcg( data ,cstep,cmin,cmax,gstep,gmin,gmax, c0, g0, acc0)
% ���ɱ�ǩ
M=size(data,1);
label=[ones(280,1);zeros(737,1)];
% ��������
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
% ��ʼֵ,���ڱȽ�
bestc = c0;
bestg = g0;
bestacc = acc0;
basenum = 2;
bestSn=0;
bestSp=0;
bestMCC=0;
% �������۷ְ�ϵ��
load(['D:\pseKNC\DHS\indices.mat'])
for x = 1:m
    for y = 1:n
        cmd = [' -c ',num2str( basenum^X(x,y) ),' -g ',num2str( basenum^Y(x,y) )];
        for i = 1:5
            %   ��ȡ���
            test = (indices == i);  %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
            train = ~test;  %train��Ԫ�صı��Ϊ��testԪ�صı��
            %   ���ݼ�
            train_data=data(train,:);   %�����ݼ��л��ֳ�train����������
            train_label=label(train,:);   %����������Ĳ���Ŀ�꣬�ڱ�������ʵ�ʷ������
            %   ���Լ�
            test_data=data(test,:); %test������
            test_label=label(test,:);
            %   Ԥ��
            model = svmtrain( train_label, train_data, cmd );
            [predict_label, accuracy, dec_values] = svmpredict(test_label, test_data,model);
            acc(1,i) = accuracy(1,1);
            [Sn_i,Sp_i,MCC_i]=perf(predict_label,test_label);
            Sn(1,i)=Sn_i;
            Sp(1,i)=Sp_i;
            MCC(1,i)=MCC_i;
            %[A,B,THRE,AUC] = perfcurve(test_label,dec_values(:,1),'1');
            %auc(1,i)=AUC;
        end
        cg(x,y)=sum(acc)/5;
        tem_Sn=sum(Sn)/5;
        tem_Sp=sum(Sp)/5;
        tem_MCC=sum(MCC)/5;
        if cg(x,y) > bestacc
            bestacc = cg(x,y);
            bestc = basenum^X(x,y);
            bestg = basenum^Y(x,y);
            bestSn=tem_Sn;
            bestSp=tem_Sp;
            bestMCC=tem_MCC;
        end
        if ( cg(x,y) == bestacc && bestc > basenum^X(x,y) )
            bestacc = cg(x,y);
            bestc = basenum^X(x,y);
            bestg = basenum^Y(x,y);
            bestSn=tem_Sn;
            bestSp=tem_SP;
            bestMCC=tem_MCC;
        end
    end
end
end
%% ����������
function [Sn,Sp,MCC]=perf(pre_label,label)
m=size(label,1);
per=[label pre_label];
TP=0;FN=0;FP=0;TN=0;
for i=1:m
    if per(i,:)==[1 1]
        TP=TP+1;end
    if per(i,:)==[1 0]
        FN=FN+1;end
    if per(i,:)==[0 1]
        FP=FP+1;end
    if per(i,:)==[0 0]
        TN=TN+1;end
end
Sn=TP/(TP+FN);
Sp=TN/(TN+FP);
MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
end
%result=[ bestSn,bestSp,bestacc,bestMCC ];