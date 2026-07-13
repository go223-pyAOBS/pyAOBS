%% this is the master job for DEM analysis

clc
clear 
close all

% basalt
k1 = 79.23; mu1 = 47.61; % Background moduli (e.g., mineral matrix)

%k1 = 73.99; mu1 = 44.34; % Background moduli (e.g., mineral matrix, carlson a lower value)
k2 = 2.32; mu2 = 0; % Inclusion moduli (e.g., water, zero shear modulus)
asp =[10000,1000,500,200,150,100,75,50,33,20]; % 1/asp=Oblate spheroids (cracks)
phic = 0.65; % Critical porosity 
rho1=2.946; %background density (e.g., mineral matrix in g/cm3)
rho2=1.03;%inclusion density (e.g., water in g/cm3)


% %Gabbro
% k1 = 88.32; mu1 = 44.01; % Background moduli (e.g., mineral matrix)
% k2 = 2.32; mu2 = 0; % Inclusion moduli (e.g., water, zero shear modulus)
% asp =[100,10,5,1]; % 1/asp=Oblate spheroids (cracks)
% %asp = 20;
% phic = 0.65; % Critical porosity
% rho1=3.0; %background density (e.g., mineral matrix in g/cm3)
% rho2=1.03;%inclusion density (e.g., water in g/cm3)
% 

% %unaltered peridotite
% k1 = 122.73; mu1 = 76.40; % Background moduli (e.g., mineral matrix)
% k2 = 2.32; mu2 = 0; % Inclusion moduli (e.g., water, zero shear modulus)
% %asp =[10000,1000,200,100,50,20]; % 1/asp=Oblate spheroids (cracks)
% asp =[100,10,5,1]; % 1/asp=Oblate spheroids (cracks)
% 
% 
% %asp = 0.1;
% phic = 0.65; % Critical porosity
% rho1=3.316; %background density (e.g., mineral matrix in g/cm3)
% rho2=1.03;%inclusion density (e.g., water in g/cm3)


% %serpentinized MIP
% k1 = 70.13; mu1 = 26.43; % Background moduli (e.g., mineral matrix)
% k2 = 2.32; mu2 = 0; % Inclusion moduli (e.g., water, zero shear modulus)
% asp =[100,10,5,1]; % 1/asp=Oblate spheroids (cracks)
% 
% %asp = 0.1;
% phic = 0.65; % Critical porosity
% rho1=2.804; %background density (e.g., mineral matrix in g/cm3)
% rho2=1.03;%inclusion density (e.g., water in g/cm3)

for j=1:length(asp)

[k, mu, por] = dem(k1, mu1, k2, mu2, 1/asp(j), phic);
rho=(1-por)*rho1+por*rho2;

% figure (1)
% subplot(1,2,1)
% plot(por, k, 'b-', por, mu, 'r--');hold on; % Plot results
% xlabel('Fractional Porosity'); ylabel('Effective Moduli');
% legend('Bulk modulus', 'Shear modulus');
% title('Elastic moduli vs Porosity')
% 
% subplot(1,2,2)
% plot(por, rho, 'k-');hold on; % Plot results
% xlabel('Fractional Porosity'); ylabel('Density (g/cm3)');
% 
% title('Density vs Porosity')

figure(1);
subplot(1,2,1); 
plot(por, k, '-', 'DisplayName', sprintf('k (asp=%5.4f)', 1/asp(j))); hold on;
plot(por, mu, '--', 'DisplayName', sprintf('mu (asp=%4.3f)', 1/asp(j))); 
xlabel('Fractional Porosity'); ylabel('Effective Moduli');
title('Elastic moduli vs Porosity');
legend('show', 'Location', 'best');  % 显示图例

subplot(1,2,2); 
plot(por, rho, '-', 'DisplayName', sprintf('rho (asp=%5.4f)', 1/asp(j))); hold on;
xlabel('Fractional Porosity'); ylabel('Density (g/cm3)');
title('Density vs Porosity');
set(gca,'xtick', 0:0.05:0.4,'ytick', 2.0:0.1:3.5,'FontSize',12,'FontName', 'Helvetica')
ylim([2.0, 3.5]);
xlim([0, 0.4]);
legend('show', 'Location', 'best');



%% calculate Vp and Vs

por_cal=0.4;

[a,~]=find(por<por_cal);

Vp=zeros(length(a),1);
Vs=zeros(length(a),1);
VpVs_ratio=zeros(length(a),1);

for i=1:length(a)

[Vp(i), Vs(i), VpVs_ratio(i)] = moduli_to_velocities(k(i,1), mu(i,1), rho(i,1));
end



figure(2);
subplot(1,3,1); 
plot(por(a), Vp, '-', 'DisplayName', sprintf('asp=%5.4f', 1/asp(j))); hold on;
xlabel('Fractional Porosity'); ylabel('Vp (km/s)');
title('Vp vs Porosity');
legend('show', 'Location', 'best');
ylim([3.0, 8.6]);
set(gca,'xtick', 0:0.05:0.4,'ytick', 3:0.2:8.6,'FontSize',12,'FontName', 'Helvetica')


subplot(1,3,2); 
plot(por(a), Vs, '-', 'DisplayName', sprintf('asp=%5.4f', 1/asp(j))); hold on;
xlabel('Fractional Porosity'); ylabel('Vs (km/s)');
title('Vs vs Porosity');
legend('show', 'Location', 'best');

subplot(1,3,3); 
plot(por(a), VpVs_ratio, '-', 'DisplayName', sprintf('asp=%5.4f', 1/asp(j))); hold on;
xlabel('Fractional Porosity'); ylabel('Vp/Vs ratio');
title('Vp/Vs vs Porosity');
legend('show', 'Location', 'best');
ylim([1.6, 2.15]);


figure(3);
plot(Vp(a), VpVs_ratio(a), 'o', 'DisplayName', sprintf('asp=%5.4f', 1/asp(j))); hold on;
xlabel('Vp (km/s)'); ylabel('Vp/Vs ratio');
title('Vp vs Vp/Vs Ratio');
legend('show', 'Location', 'best');
xlim([3, 8.5]); ylim([1.6, 2.15]);

end

%fprintf('Vp = %.2f km/s, Vs = %.2f km/s, Vp/Vs = %.2f\n', Vp, Vs, VpVs_ratio);