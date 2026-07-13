function [Vp, Vs, VpVs_ratio] = moduli_to_velocities(K, mu, rho)
    % Inputs:
    %   K, mu: Bulk and shear moduli (GPa)
    %   rho: Density (g/cm³)
    % Outputs:
    %   Vp, Vs: Velocities in km/s
    %   VpVs_ratio: Dimensionless ratio
    
    % Convert density to kg/m³ and moduli to Pa
    rho = rho * 1000;          % g/cm³ → kg/m³
    K = K * 1e9;               % GPa → Pa
    mu = mu * 1e9;             % GPa → Pa
    
    % Calculate velocities (m/s)
    Vp = sqrt((K + (4/3) * mu) / rho);
    Vs = sqrt(mu / rho);
    
    % Convert to km/s
    Vp = Vp / 1000;
    Vs = Vs / 1000;

    VpVs_ratio = safeDivide(Vp, Vs);
    
    
end

function result = safeDivide(numerator, denominator)
    if denominator <= 0.0001
        result = NaN; % or Inf, or whatever makes sense for your application
        warning('Division by zero attempted');
    else
        result = numerator / denominator;
    end
end
