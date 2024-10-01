# SUMO学习笔记

- SUMO仿真环境运行需要三个文件Network、Route以及SUMO configuration file。
  - Network(.net.xml)：路网创建、交通信号灯创建

    - 路网文件生成方式
      - 外部导入，如OSM等
        - `netconvert --osm-files osm.osm -o net.net.xml`
      - netedit编辑得到：使用界面进行编辑得到，所有均在可视化界面中完成
      - 人工定义得到需要人工定义边、节点、连接以及信号、配时等
        - `netconvert --node-files=nodes.nod.xml --edge-files=edges.edg.xml --type-files=types.typ.xml --output-file=net.net.xml`

  - Route(.rou.xml)：创建车流信息

    - rou文件的基本结构如下：
      - XML头部
      - routes元素
      - 车辆类型定义
      - 路由定义
      - 车辆定义

    - XML头部：rou文件是一个XML文件,因此需要以XML声明开始：

      ```xml
      <?xml version="1.0" encoding="UTF-8"?>
      ```

    - routes元素：整个文件的内容都包含在<routes>标签内：

        ```xml
        <routes>
            <!-- 其他元素将在这里定义 -->
        </routes>
        ```

    - 车辆类型定义：使用<vType>元素定义车辆类型,可以指定诸如长度、最大速度、加速度等属性：

        ```xml
        <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="55.56" color="1,0,0"/>
        ```

        - `id`：定义车辆类型为唯一标识符
        - `vclass`：车辆类型
            - passenger：乘用车
            - bus：公交车
            - truck：卡车
            - motorcycle：摩托车
            - bicycle：自行车
            - emergency：紧急车辆
            - evehicle：电动车
            - pedestrian：行人
        - `accel`：车辆的最大加速度为
        - `decel`：车辆的最大减速度为
        - `sigma`：驾驶员行为的随机性系数
        - `length`：车辆长度 米
        - `width`：车辆宽度 米
        - `maxSpeed`：车辆最大速度 m/s
        - `minGap`：车辆之间保持的最小安全距离 米
        - `tau`：驾驶员的反应时间 秒
        - `color`：车辆颜色（RGB值，范围0-1）
        - `laneChangeModel`：车辆使用的变道模型
            - LC2013（Lane Change 2013）：默认的变道模型，结合了变道的安全性和期望性，适合一般的城市交通仿真。
            - SL2015（Safe Lane Change 2015）：在 LC2013 的基础上进行扩展，专注于改进变道安全性和避免碰撞的情况。适合在更复杂的交通场景或需要高度安全驾驶行为的场景下使用。
        - `carFollowModel`：车辆使用的跟车模型
            - Krauss 模型：适合常规情况，随机性较强
            - IDM 模型：适合更精确的仿真，注重物理和动力学特性
            - Wiedemann 模型：适合复杂的驾驶行为分析
            - Bando 模型：适合低速和拥堵情况下的研究
        - imperfection：驾驶员的不完美系数为 0.1，表示驾驶员相对较稳定

    - 路由定义：使用<route>元素定义路由,指定车辆将要经过的边的ID序列：

      ```xml
      <route id="route0" edges="edge1 edge2 edge3"/>
      ```

      - `id`：路线的唯一标识符。

      - `edges`：定义行驶路线的边，使用边的 ID，用空格分隔（这些 ID 必须与网络文件中的道路 ID 一致）。
      - `color`：定义路线的颜色（用于可视化）。

    - 车辆定义：使用<vehicle>或<flow>元素定义具体的车辆：

        ```xml
        <vehicle id="veh0" type="car" route="route0" depart="0"/>
        ```

        - `id`：车辆的唯一标识符。

        - `type`：指定车辆类型（与 `<vType>` 中定义的车辆类型相对应）。

        - `route`：指定车辆的行驶路线（与 `<route>` 中定义的路线相对应）。

        - `depart`：车辆的发车时间（以秒为单位）。

        - `departLane`：车辆初始进入的车道，默认值是 `best`（即由仿真决定最佳车道）。

        - `departPos`：车辆初始进入的道路位置（以米为单位，默认值为0，表示从道路的起点开始）。

        - `departSpeed`：车辆初始的速度（默认值是 0 米/秒）。

        - `arrivalLane`：车辆到达目的地时的车道选择，默认值是 `current`（保持当前车道）。

        - `arrivalPos`：车辆到达目的地时的位置。

        - `arrivalSpeed`：车辆到达时的速度。

        - `color`：定义车辆的颜色。

    - 或者定义车流：

        ```xml
        <flow id="flow1" type="car" route="route0" begin="0" end="3600" vehsPerHour="600" departLane="best" departSpeed="max"/>
        ```

        - `id`：流量的唯一标识符。（必须）
        - `type`：流量中车辆的类型（对应 `<vType>` 元素）。（必须）

        - `route`：流量中车辆的行驶路线（对应 `<route>` 元素）。（必须）

        - `begin`：流量开始的时间（以秒为单位）。（必须）

        - `end`：流量结束的时间（以秒为单位）。

        - `departLane`：车辆生成时所在的车道（默认值为 `best`，表示由仿真选择最佳车道）。

        - `departSpeed`：车辆生成时的速度。

        - `vehsPerHour`：流量中每小时生成的车辆数量。

        - `number`：总生成的车辆数量。

        - `probability`：生成车辆的概率（范围为 0 到 1，表示在每个时间步长生成车辆的概率）。

        - `period`：生成车辆的时间周期（与 `probability` 互斥，用来替代生成概率以时间间隔定义车辆生成）。

    - 完整的rou文件示例：

        ```xml
        <?xml version="1.0" encoding="UTF-8"?>
        <routes>
            <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="55.56" color="1,0,0"/>
            <route id="route0" edges="edge1 edge2 edge3"/>
            <vehicle id="veh0" type="car" route="route0" depart="0"/>
            <flow id="flow0" type="car" route="route0" begin="0" end="3600" vehsPerHour="300"/>
        </routes>
        ```

    - 其他可能的元素和属性：

        - 随机路由：

          ```xml
          <routeDistribution id="routeDist1">
              <route id="route1" edges="edge1 edge2" probability="0.7"/>
              <route id="route2" edges="edge1 edge3" probability="0.3"/>
          </routeDistribution>
          ```

        - 停靠点：

          ```xml
          <stop lane="edge1_0" duration="30"/>
          ```

        - 车辆参数分布：

          ```xml
          <param key="has.rerouting.device" value="true"/>
          ```

        - 编写rou文件时,需要注意：
          - 确保所有引用的边(edge)在网络文件中存在
          - 车辆ID和路由ID必须唯一
          - 出发时间(depart)必须非负
          - 确保路由是连续的,即每条边都与下一条边相连

  - SUMO configuration file(.sumocfg)：将上述两个文件结合，实现仿真

    - sumocfg文件的基本结构如下：
      - XML头部
      - configuration元素
      - input部分
      - output部分
      - time部分
      - processing部分
      - routing部分
      - report部分
      - gui_only部分
    - XML头部： sumocfg文件是一个XML文件，因此需要以XML声明开始：


    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    ```

    - configuration元素：整个文件的内容都包含在<configuration>标签内：


    ```xml
    <configuration>
        <!-- 其他元素将在这里定义 -->
    </configuration>
    ```

    - input部分：在这部分定义仿真所需的输入文件：


    ```xml
    <input>
        <net-file value="mynetwork.net.xml"/>
        <route-files value="myroutes.rou.xml"/>
        <additional-files value="additional.xml"/>
    </input>
    ```

    - net-file：指定网络文件
    - route-files：指定路由文件
    - additional-files：指定其他额外文件，如交通灯计划、检测器等

    - output部分：定义仿真的输出文件：

        ```xml
        <output>
            <tripinfo-output value="tripinfo.xml"/>
            <summary-output value="summary.xml"/>
        </output>
        ```

        - tripinfo-output：输出车辆行程信息
        - summary-output：输出仿真摘要信息

    - time部分：定义仿真的时间设置：
        ```xml
        <time>
            <begin value="0"/>
            <end value="3600"/>
            <step-length value="0.1"/>
        </time>
        ```

        - begin：仿真开始时间

        - end：仿真结束时间

        - step-length：仿真步长

    - processing部分： 定义仿真的处理选项：

        ```xml
        <processing>
            <ignore-route-errors value="true"/>
            <time-to-teleport value="300"/>
            <max-depart-delay value="900"/>
        </processing>
        ```

        - ignore-route-errors：是否忽略路由错误

        - time-to-teleport：车辆被卡住多长时间后进行传送

        - max-depart-delay：车辆最大出发延迟时间

    - routing部分：定义路由相关选项：
        ```xml
        <routing>
            <device.rerouting.probability value="0.8"/>
            <device.rerouting.period value="300"/>
        </routing>
        ```

        - device.rerouting.probability：车辆重新路由的概率

        - device.rerouting.period：重新路由的时间间隔

    - report部分：定义报告和日志相关选项：

        ```xml
        <report>
            <verbose value="true"/>
            <no-step-log value="false"/>
        </report>
        ```

        - verbose：是否输出详细信息

        - no-step-log：是否禁用步骤日志

    - gui_only部分：定义GUI相关选项（仅在sumo-gui中有效）：

        ```xml
        <gui_only>
            <start value="true"/>
            <quit-on-end value="true"/>
        </gui_only>
        ```

        - start：是否在启动时立即开始仿真

        - quit-on-end：仿真结束时是否自动退出

    - 完整的sumocfg文件示例：

        ```xml
        <?xml version="1.0" encoding="UTF-8"?>
        <configuration>
            <input>
                <net-file value="mynetwork.net.xml"/>
                <route-files value="myroutes.rou.xml"/>
                <additional-files value="additional.xml"/>
            </input>
        
            <output>
                <tripinfo-output value="tripinfo.xml"/>
                <summary-output value="summary.xml"/>
            </output>
        
            <time>
                <begin value="0"/>
                <end value="3600"/>
                <step-length value="0.1"/>
            </time>
        
            <processing>
                <ignore-route-errors value="true"/>
                <time-to-teleport value="300"/>
                <max-depart-delay value="900"/>
            </processing>
        
            <routing>
                <device.rerouting.probability value="0.8"/>
                <device.rerouting.period value="300"/>
            </routing>
        
            <report>
                <verbose value="true"/>
                <no-step-log value="false"/>
            </report>
        
            <gui_only>
                <start value="true"/>
                <quit-on-end value="true"/>
            </gui_only>
        </configuration>
        ```

    - 编写sumocfg文件时，需要注意：
      - 确保所有引用的文件（如网络文件、路由文件等）都存在且路径正确。
      - 时间设置要合理，确保end时间大于begin时间。
      - 根据具体需求调整处理选项和输出选项。
      - 如果使用sumo-gui，可以添加gui_only部分来定制GUI行为。

- 