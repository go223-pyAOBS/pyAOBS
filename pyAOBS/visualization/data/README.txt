请将全球/区域重力异常格网命名为 gravity_world.grd 放置在本目录，
或通过「重力工具箱」中的数据来源目录改用其它路径。

推荐使用 NetCDF GMT 格式的 .grd（可由 xarray 读取），单位为 mGal 的 Bouguer /
自由空气等需与你模型异常类型自行一致。

包内默认路径等价于导入路径 pyAOBS.visualization.data （历史命名亦有 pyobs.visualization.data 所指含义）。

Qt「重力工具箱」中按钮「平面图 lon×lat」会读取本格网，在经纬度平面上叠加快模型剖面轨迹与观测站位。
