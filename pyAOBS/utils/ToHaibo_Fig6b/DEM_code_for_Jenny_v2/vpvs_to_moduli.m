function [K_eff, Mu_eff] = vpvs_to_moduli(Vp, Vs, rho)
    % Convert Vp and Vs (km/s) to effective moduli (GPa)
    % Inputs:
    %   Vp, Vs: P-wave and S-wave velocities (km/s)
    %   rho: Density (g/cm³)
    % Outputs:
    %   K_eff: Bulk modulus (GPa)
    %   Mu_eff: Shear modulus (GPa)
    
    % Convert km/s to m/s and g/cm³ to kg/m³ for SI units
    Vp = Vp * 1000;      % km/s → m/s
    Vs = Vs * 1000;      % km/s → m/s
    rho = rho * 1000;    % g/cm³ → kg/m³
    
    % Calculate moduli (in Pascals)
    Mu_eff = rho * Vs^2;                     % Shear modulus (Pa)
    K_eff = rho * (Vp^2 - (4/3) * Vs^2);     % Bulk modulus (Pa)
    
    % Convert to GPa
    K_eff = K_eff / 1e9;
    Mu_eff = Mu_eff / 1e9;
end