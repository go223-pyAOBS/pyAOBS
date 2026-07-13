%[K_eff, Mu_eff] = vpvs_to_moduli(6.96, 4.02, 2.946); % basalt (Carlson 2014 Gcube, doi:10.1002/2014GC005537.)

%[K_eff, Mu_eff] = vpvs_to_moduli(7.00, 3.83, 3.0);% Gabbro (Gregory 2021 Geology,https://doi.org/10.1130/G49097.1)

[K_eff, Mu_eff] = vpvs_to_moduli(8.23, 4.80, 3.316);% unalterd peridotite (Gregory 2021 Geology,https://doi.org/10.1130/G49097.1)


fprintf('Bulk Modulus (K_eff) = %.2f GPa\n', K_eff);
fprintf('Shear Modulus (μ_eff) = %.2f GPa\n', Mu_eff);