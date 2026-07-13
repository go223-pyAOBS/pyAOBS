#！/bin/bash matlab是之前测试的时候的程序，把相应的数据弄好之后我又在gmt里面画最终的图Fig6b

gmt set FONT 8p,Helvetica,black
gmt set FONT_LABEL 8p,Helvetica,black
gmt set FONT_ANNOT_PRIMARY 8p,Helvetica,black
gmt set MAP_FRAME_TYPE=plain
gmt set COLOR_BACKGROUND=white
gmt set FORMAT_GEO_MAP ddd:mm:ss #ddd.xx
gmt set PAPER_MEDIA A4

# plot the lithology categories


fig_name1=Fracture_water_dot_graph

fig_fmt=pdf # pdf for vector diagram

R=3000/8550/1.60/2.15
J=X7./10

gmtsoff=3000
gmtnoff=8550

gmt begin $fig_name1 $fig_fmt
gmt basemap -R$R -J$J -Bxa500+l"Vp (m/s)" -Bya0.05+l"Vp/Vs" -BnesW
gmt basemap -R3.0/8.550/1.60/2.15 -J$J -Bxa1g0.5+l"Vp (km/s)" -Bya0.05g0.05+l"Vp/Vs" -BneSW

echo $gmtsoff '1.9\n' $gmtnoff '1.9' | gmt plot -R$R -J$J -W2p,79 # this is the average water content
# # Carlson and miller partilly serpentinized
# echo "4552 2.2 " > zzz
# echo "8299 1.754" >> zzz
# gmt psxy zzz -R$R -J$J -W12,201/202/202 
# # Serpentinite - Horen
# cat heron.dat | awk '{print ($2 " " $4 "\n")}' > zzz
# gmt psxy zzz -R$R -J$J -W12,89/87/87 
#data model

#画背景中灰色data点
awk 'NR%5==1{print $1,$2}'  model_data.txt | gmt psxy -R$R -J$J -Sc0.1 -Ggray 

#画Vp/Vs> 1.9 的属于每个多边形内的点 （背景彩色的点）
awk '{print $1,$2}' ./Polygon_1_points.txt | gmt psxy -R$R -J$J -Sc0.1 -G0/255/127.5
awk '{print $1,$2}' ./Polygon_2_points.txt | gmt psxy -R$R -J$J -Sc0.1 -G0/223.125/143.4375
awk '{print $1,$2}' ./Polygon_3_points.txt | gmt psxy -R$R -J$J -Sc0.1 -G0/191.25/159.375
awk '{print $1,$2}' ./Polygon_4_points.txt | gmt psxy -R$R -J$J -Sc0.1 -G0/159.375/175.3125
awk '{print $1,$2}' ./Polygon_5_points.txt | gmt psxy -R$R -J$J -Sc0.1 -G0/127.5/191.25

#plot aspect ratio lines画选取不同aspect ratio情况下的Vp/Vs vs Vp趋势线（图中黑色细线）
#gmt psxy zzz -R$R -J$J -St0.4 -W0.1p,black -Gblack 
awk '{print $2*1000,$3}' ./1asp0.0001.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./2asp0.001.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./3asp0.002.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./4asp0.005.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./5asp0.0067.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./6asp0.01.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./7asp0.013.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./8asp0.02.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./9asp0.03.txt | gmt psxy -R$R -J$J -W0.7p,black
awk '{print $2*1000,$3}' ./10asp0.05.txt | gmt psxy -R$R -J$J -W0.7p,black

# plot polygon画彩色多边形（多边形的边是相同孔隙度）
awk '{print $1*1000,$2}' ./Polygon_1.txt | gmt plot -R$R -J$J -t70 -W1p,0/255/127.5 -G0/255/127.5
awk '{print $1*1000,$2}' ./Polygon_2.txt | gmt plot -R$R -J$J -t70 -W1p,0/223.125/143.4375 -G0/223.125/143.4375
awk '{print $1*1000,$2}' ./Polygon_3.txt | gmt plot -R$R -J$J -t70 -W1p,0/191.25/159.375 -G0/191.25/159.375
awk '{print $1*1000,$2}' ./Polygon_4.txt | gmt plot -R$R -J$J -t70 -W1p,0/159.375/175.3125 -G0/159.375/175.3125
awk '{print $1*1000,$2}' ./Polygon_5.txt | gmt plot -R$R -J$J -t70 -W1p,0/127.5/191.25 -G0/127.5/191.25
awk '{print $1*1000,$2}' ./Polygon_6.txt | gmt plot -R$R -J$J -t70 -W1p,0/95.625/207.1875 -G0/95.625/207.1875
awk '{print $1*1000,$2}' ./Polygon_7.txt | gmt plot -R$R -J$J -t70 -W1p,0/63.75/223.125 -G0/63.75/223.125
awk '{print $1*1000,$2}' ./Polygon_8.txt | gmt plot -R$R -J$J -t70 -W1p,0/31.875/239.0625 -G0/31.875/239.0625
awk '{print $1*1000,$2}' ./Polygon_9.txt | gmt plot -R$R -J$J -t70 -W1p,0/0/255 -G0/0/255


# Dunite - christensen 1996  （图中黄色的dunite点）
echo " 8.299 1.754 0.091 0.045" > zzz
cat zzz | awk '{print ($1*1000. " " $2 "\n")}' > z_DUN.xy
cat zzz | awk '{print (($1+$3/2)*1000. " " $2 "\n" ($1-$3/2)*1000. " " $2 "\n")}' > z_DUN_sd1.xy
cat zzz | awk '{print ($1*1000. " " ($2+$4/2) "\n" $1*1000. " " ($2-$4/2) "\n")}' > z_DUN_sd2.xy

gmt psxy z_DUN_sd1.xy -R$R -J$J -W3 -Gblack
gmt psxy z_DUN_sd2.xy -R$R -J$J -W3,black 
gmt psxy z_DUN.xy -R$R -J$J -Sc0.35 -Gblack 
gmt psxy z_DUN.xy -R$R -J$J -Sc0.3 -Gyellow 

gmt end show 

\rm z*

