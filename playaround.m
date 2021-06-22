close all; clc; clear;
BIG=[];
for i=0:7
    big=load("BIG"+i+".txt");
    block=reshape(big,384,12);
    %figure(i+1)
    %spy(block)
    BIG=[BIG;block];
    
end


i=1;
j=1;
while i<3072 
    if BIG(i)~=0;
        a(j)=BIG(i);
        j=j+1;
    end
    i=i+1;
end

histogram(a)

% load("/home/jimenez/Matlab_scripts/LQCD_A1.mat")
% %load('LQCD_A2.mat')
% dimensions= size(A1);
% n=dimensions(1);
% m0 =-0.1;
% A1=A1+m0*speye(n,n);
%
% s=12;
% Nb= n/s;
%
% inv_A=load("inv_m0_01.txt");
% N=1000;
% measurements=zeros(s,s,N);
%
%
% iden = eye(s,s);
% for i=1:N
%       measurements(:,:,i)= BIG'*inv_A*BIG;%%inv_A*X;
%     i
% end
%
%
% block_trace=0;
% for i=1:N
%     block_trace = block_trace + measurements(:,:,i);
% end
% block_trace= block_trace/N
%
%
%
% non_shifted = trace(block_trace)
%





%% C-CODE for printing
% char* a[100] ;
%     sprintf(a, "%s%d%s", "BIG",rank, ".txt");
%     char fileSpec[strlen(a)+1];
%     snprintf( fileSpec, sizeof( fileSpec ), "%s", a );
%    FILE * fp;
%    fp = fopen (fileSpec, "w+");
%    for(j=0; j<block_size; j++){
% 		for(i=0; i<l->inner_vector_size; i++){
% 		   fprintf(fp, "%f\n", j,i, creal(X[j][i]));
%    fclose(fp);
%
%   END_MASTER(threading)
%   SYNC_MASTER_TO_ALL(threading)