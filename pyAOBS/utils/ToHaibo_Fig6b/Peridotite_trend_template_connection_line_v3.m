%% this is fracture water template for Peridotite

clc;
clear;
close all;

% 定义文件列表   先用DEM算出不同aspect ratio下的Vp vs Vp/Vs 曲线，和对应的孔隙度
files = {
'1asp0.0001.txt','2asp0.001.txt',...    
'3asp0.002.txt', '4asp0.005.txt', '5asp0.0067.txt', ...
    '6asp0.01.txt', '7asp0.013.txt', '8asp0.02.txt', '9asp0.03.txt', '10asp0.05.txt'
};
num_files = length(files);
asp=[0.0001,0.001,0.002,0.005,0.0067,0.01,0.013,0.02,0.03,0.05];

% 读取所有数据
data = cell(num_files, 1);
ref_porosity = zeros(num_files, 1);
for i = 1:num_files
    data{i} = load(files{i});
    ref_porosity(i) = data{i}(end, 1); % 获取每个文件最后一行的孔隙度
end

% 创建主图
figure;
hold on;
grid on;
xlabel('Vp (km/s)');
ylabel('Vp/Vs');
set(gca,'xtick', 2:0.5:8.5,'ytick', 1.6:0.05:2.15,'FontSize',12,'FontName', 'Helvetica')
xlim([2, 8.5]); ylim([1.6, 2.15]);
%title('Vp-Vp/Vs plot with connected porosity points for different aspect ratio and scatter data points');

% 3. 添加散点数据 (第三列/1000为x轴，第五列为y轴)，只选择第五列>1.6的点
data_file = 'model_data.txt';
scatter_data = load(data_file);
x_scatter = scatter_data(:,1)/1000; % 第三列除以1000作为x轴 Vp
y_scatter = scatter_data(:,2);     % 第五列作为y轴 Vp/Vs

% 筛选第五列>1.6的点
selected_idx = y_scatter > 1.6;
x_selected = x_scatter(selected_idx);
y_selected = y_scatter(selected_idx);

% 每五个点画一个点
x_selected = x_selected(1:5:end);
y_selected = y_selected(1:5:end);

% 绘制筛选后的散点图
scatter(x_selected, y_selected, 50, 'MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor',[0.5 0.5 0.5], 'DisplayName', 'Data points');

% 颜色设置
curve_colors = lines(num_files); % 原始曲线颜色
conn_colors = jet(num_files-1); % 连接线颜色

% 1. 先绘制所有原始曲线(Vs ≤ 2.15)
for i = 1:num_files
    valid_idx = data{i}(:,3) <= 2.17;
    % plot(data{i}(valid_idx,2), data{i}(valid_idx,3), ...
    %      'k', 'LineWidth', 1.5, ...
    %     'DisplayName', [files{i}]);
   %plot(data{i}(valid_idx,2), data{i}(valid_idx,3), 'k', ...
   % 'DisplayName', sprintf('ASP=%.4f', asp(1,i)), ...
   % 'LineWidth', 1.5);
   plot(data{i}(valid_idx,2), data{i}(valid_idx,3), 'k', 'LineWidth', 1.5);

end

% 2. 绘制连接线
for start_file = 1:num_files-1
    % 获取起始文件的参考孔隙度
    current_ref_poro = ref_porosity(start_file);
    
    % 收集从start_file到最后一个文件的数据点
    Vp_points = [];
    Vs_points = [];
    file_indices = [];
    
    for file_idx = start_file:num_files
        % 在当前文件中寻找最接近参考孔隙度的点(Vs ≤ 2.17)
        current_data = data{file_idx};
        valid_idx = current_data(:,3) <= 2.17;
        
        if any(valid_idx)
            [~, closest_idx] = min(abs(current_data(valid_idx,1) - current_ref_poro));
            selected_idx = find(valid_idx, closest_idx);
            selected_idx = selected_idx(end); % 获取最后一个满足条件的索引
            
            Vp_points = [Vp_points; current_data(selected_idx,2)];
            Vs_points = [Vs_points; current_data(selected_idx,3)];
            file_indices = [file_indices; file_idx];
        end
    end
    
    % 按Vp排序后连接
    [sorted_Vp, sort_idx] = sort(Vp_points);
    sorted_Vs = Vs_points(sort_idx);
    sorted_files = file_indices(sort_idx);

    % 输出连接线信息
fprintf('\nConnection Line %d (Start File: %s, Reference Porosity: %.4f):\n', ...
    start_file, files{start_file}, current_ref_poro);
fprintf('%-10s %-10s %-15s\n', 'File', 'Vp', 'Vs');
for i = 1:length(sorted_Vp)
    fprintf('%-10s %-10.4f %-10.4f\n', ...
        files{sorted_files(i)}, sorted_Vp(i), sorted_Vs(i));
end

    
    % %绘制连接线
    if length(sorted_Vp) >= 2
        plot(sorted_Vp, sorted_Vs, '-o', 'Color', conn_colors(start_file,:), ...
            'MarkerFaceColor', conn_colors(start_file,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('Fractional porosity (φ=%.4f)', current_ref_poro));
    end
end

% 添加七个多边形并填充 这部分需要根据上一部分计算的 connection line （也就是不同asp下相同的porosity构成多边形两条边，多边形的底部的边就是asp=0.05的部分数据）
polygon_files = {
    'Polygon_1.txt', 'Polygon_2.txt', 'Polygon_3.txt', ...
    'Polygon_4.txt', 'Polygon_5.txt', 'Polygon_6.txt', 'Polygon_7.txt', 'Polygon_8.txt','Polygon_9.txt'};
wt_percentages = [0.04, 0.14, 0.45, 0.89, ...
                  1.46, 2.25, 3.46, 5.42, 8.59]; % 这一部分是每个多边形的内的fracture water
poly_colors = winter(9); % 为多边形选择不同的颜色
poly_colors =flip(poly_colors);
for i = 1:9
    poly_data = load(polygon_files{i});
    fill(poly_data(:,1), poly_data(:,2), poly_colors(i,:), 'FaceAlpha', 0.3, ...
        'EdgeColor', poly_colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Pore-water wt%%=%.2f', wt_percentages(i)));

    % 添加标签
    [centroid_x, centroid_y] = calculateCentroid(poly_data(:,1), poly_data(:,2));
    text(centroid_x, centroid_y, sprintf('%.2f%%', wt_percentages(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% 添加图例并调整布局
legend('Location', 'bestoutside');
hold off;

% 辅助函数：计算多边形质心
function [centroid_x, centroid_y] = calculateCentroid(x, y)
    % 计算多边形质心
    x = x(:);
    y = y(:);
    n = length(x);
    x = [x; x(1)];
    y = [y; y(1)];
    
    area = 0;
    centroid_x = 0;
    centroid_y = 0;
    
    for i = 1:n
        cross_prod = x(i)*y(i+1) - x(i+1)*y(i);
        area = area + cross_prod;
        centroid_x = centroid_x + (x(i) + x(i+1)) * cross_prod;
        centroid_y = centroid_y + (y(i) + y(i+1)) * cross_prod;
    end
    
    area = area / 2;
    centroid_x = centroid_x / (6*area);
    centroid_y = centroid_y / (6*area);
end

disp('Plot completed with all polygons added and scatter points thinned to every 5th point.');




% ... [previous plotting code remains unchanged] ...

% Set up A4 paper size settings
set(gcf, 'Units', 'centimeters');  % Use centimeter units
set(gcf, 'PaperUnits', 'centimeters');
%set(gcf, 'PaperSize', [21 29.7]);  % A4 size (21×29.7 cm)
set(gcf, 'PaperPositionMode', 'manual');
%set(gcf, 'PaperPosition', [0 0 21 29.7]);  % Full page
set(gcf, 'Color', 'w');  % White background


% Adjust figure size to maintain aspect ratio (optional)
fig_width = 5;  % Width in cm (leaving margins)
fig_height = fig_width * 29.7/21;  % Maintain A4 aspect ratio
set(gcf, 'Position', [10 10 fig_width fig_height]);

% Save as vector graphic (choose one format)
saveas(gcf, 'Vp_Vs_Plot_A4.eps', 'epsc');  % Recommended for publications






