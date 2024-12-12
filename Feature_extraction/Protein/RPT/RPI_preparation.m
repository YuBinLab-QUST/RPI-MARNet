clear all
clc
fid=fopen('RPI488_protein.fa');%读取序列数据，计算序列数量
string=fscanf(fid,'%s'); %从fid文件读取一行数据，存储在string中，%s表示读取字符串类型数据
%匹配的字符串
firstmatches=findstr(string,'>')+6;%查找string中第一个'>'出现的位置，将其加6后，获得序列的开始位置
endmatches=findstr(string,'>')-1;%结束位置
firstnum=length(firstmatches); %firstnum=endnum序列的条数
endnum=length(endmatches);

Dim_number=67

  for k=1:Dim_number
    j=1;
    jj=1;
    lensec(k)=endmatches(k+1)-firstmatches(k)+1;%每条序列的长度
    for nnn=1:2%循环两次，用于填充序列的前缀和后缀
    sign=[ '>',num2str(k)];%生成一个表示序列编号的字符串，如>1等
    sequ(2*k-1,jj)=sign(nnn);%将生成的符号填充到序列中。`jj`用于记录当前序列中已填充的符号数量
    jj=jj+1;
    end
   for mm=firstmatches(k):endmatches(k+1)%遍历当前序列中每个匹配字符
        sequ(2*k,j)=string(mm); %从匹配位置提取字符，并填充到序列中
        j=j+1;%更新已填充字符的索引
   end
  end

   x=cellstr(sequ);%将变量sequ转换为字符串类型
RPI_protein_N_lensec=lensec

WEISHU=488  #自己加
for i=1:WEISHU    %WEISHU在PsePSSM.m外加中
    nnn=num2str(i);%将i转换为字符串
    name = strcat(nnn,'.pssm');%拼接字符串以创建文件名，格式为i.pssm
    fid{i}=importdata(name);%importdata读取文件
end

C={};
for t=1:WEISHU
    shu=fid{t}.data;
    shuju=shu(1:RPI_protein_N_lensec(1,t),1:20);
    RPI_protein_N_PSSM{t}=shuju;%将文件内容赋值给字典C的相应键值对中，字典键的文件名即RPI_protein_N_PSSM{t}
end

pssm=[];
maxlen=[];
for i=1:numel(RPI_protein_N_PSSM)  %遍历包含多个PSSM文件的数组
    data=RPI_protein_N_PSSM{i};
    data= 1.0 ./ ( 1.0 + exp(-data) );%用指数函数和倒数运算得到一个新的数据
    pssm=[pssm;data];  %%%%把所有的PSSM文件前L行20列保存到一个文件里
    [row,column]=size(data);
    maxlen=[maxlen;row];
    data=[];%消除临时数据和行数
    row=[];
end
index_PA=cumsum(maxlen);   %%%cumsum函数通常用于计算一个数组各行的累加值，index_PA得到所有的行数
maxlen=[];
index_PA=[0;index_PA];%只保留index_PA的最后一列，作为最终的行数索引
[m,n]=size(index_PA);
save tpc1_python_PSSM.mat pssm index_PA 

%将 `pssm`、`index_PA` 数组以及一些附加信息（如行数和列数）保存到名为 `tpc1_python_PSSM.mat` 的矩阵文件中
%tpc1_python_PSSM.mat文件在RPT_creat.py中使用



