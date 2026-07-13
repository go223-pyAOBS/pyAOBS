% \u6570\u636E\u8BFB\u53D6\uFF1A\u598245B4BDDC.525\u6587\u4EF6\u7684\u8BFB\u53D6\uFF1A
fid=fopen('45B4BDDC.525');
OBEM_data=fread(fid,[5,inf],'int32');
OBEM_data = OBEM_data';
Mz1=OBEM_data(:,1);% \u5782\u76F4Z\u5411\u78C1\u9053\u78C1\u573A\u503C
My1=OBEM_data(:,2);% \u6C34\u5E73Y\u65B9\u5411\u78C1\u9053\u78C1\u573A\u503C
Mx1=OBEM_data(:,3);% \u6C34\u5E73X\u65B9\u5411\u78C1\u9053\u78C1\u573A\u503C
E11=OBEM_data(:,4);% \u7B2C\u4E00\u7535\u9053\u7535\u4F4D\u5DEE
E21=OBEM_data(:,5);% \u7B2C\u4E8C\u7535\u9053\u7535\u4F4D\u5DEE

% \u6570\u636E\u6821\u6B63
%\u5728\u8BFB\u53D6\u6570\u636E\u540E\uFF0C\u6839\u636E\u8F6C\u6362\u53C2\u6570\u8FDB\u884C\u6570\u636E\u6821\u6B63\uFF0C\u5F97\u5230nV\u548CnT
Mx=0.00004218*Mx1;%Mx,chanel 3 of obem's data,change unit to nT
My=0.00004218*My1;%My,chanel 2 of obem's data,change unit to nT
Mz=0.00004218*Mz1;%Mz,chanel 1 of obem's data,change unit to nT
E1=0.95*E11/1650;%E1,chanel 4 of obem's data,change unit to nV
E2=0.95*E21/1650;%E2,chanel 5 of obem's data,change unit to nV

% \u6570\u636E\u5B58\u50A8
save -ascii Bz.txt Mz;
save -ascii By.txt My;
save -ascii Bx.txt Mx;
save -ascii E1.txt E1;
save -ascii E2.txt E2;