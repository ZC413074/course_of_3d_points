**ubuntu1804LTS run [PointRCNN](https://github.com/sshaoshuai/PointRCNN) 预训练模型 踩雷记录**
<br/>    

## 一、安装NVIDIA驱动、cuda、cudnn
1. 在nvidia官网根据自己的设别下载对应版本的[NVIDIA驱动](https://www.nvidia.com/Download/index.aspx?lang=en-us)、[cuda](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu)、[cudnn](https://developer.nvidia.com/rdp/cudnn-downloads) 例如：64位pc、RTX2080 -> cuda10.2   
    1)[cuda版本选择](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#abstract)
    ```
    lshw -numeric -C display  ///>   查看显卡型号
    lspci | grep -i nvidia    ///> 或 查看本机显卡
    ```
2. 安装nvidia   
   1）旧驱动卸载以及禁用自带的 nouveau 
   ```
   sudo apt-get purge nvidia*                       ///>   卸载旧的驱动   
   sudo apt-get autoremove
   sudo vi /etc/modprobe.d/blacklist-nouveau.conf   ///>  禁用自带的 nouveau nvidia驱动
   ```  
   加入一下内容：   
   ```
   blacklist nouveau 
   options nouveau modeset=0
   ```
   ```
   sudo update-initramfs -u    ///> 更新一下
   lsmod | grep nouveau        ///> 无输出表示禁用生效
   ```    
   <br/> 
   2）显示器设置  
   台式pc:将显示器接口插入到主机集成显卡接口    
   笔记本pc:进入bios目录，关掉独立显卡显示    

   3）进入系统文本编辑
   ```
   sudo telinit 3
   ```

   4）安装驱动    
   ```
   sudo chmod a+x NVIDIA-Linux-x86_64-450.80.02.run       ///>  切换到驱动文件路径下，并给驱动文件增加可执行权限
   ./NVIDIA-Linux-x86_64-450.80.02.run --no-opengl-files  ///>  执行  默认执行即可
   nvidia-smi
   ```     
   显示如下：
    ![](https://i.loli.net/2021/04/26/olaTjEW67PCDHuN.png)      
    <br/>     
3. 安装cuda    
   1）卸载旧版本的cuda 
   ```
   sudo /usr/local/cuda-8.0/bin/cuda_uninstall    ///> 根据老版本的cuda名卸载
   sudo rm -rf /usr/local/cuda-8.0
   ```
   2）安装新版的cuda    
   ```
   sudo sh cuda_10.0.130_410.48_linux.run        ///> 安装cuda
   sudo /etc/init.d/lightdm start                ///> 安装好后，打开图形界面     
   sudo gedit ~/.bashrc                          ///> 更改环境变量
   ```
   添加如下内容：
   ```
    export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}       
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}   
    export CUDA_HOME=/usr/local/cuda-10.0
   ```
   ```
   source ~/.bashrc    ///> 环境变量生效
   nvcc -V             ///>  查看cuda是否安装成功
   ```

4. 安装cudnn     
   ```
   sudo dpkg -i  libcudnn8-samples_ 
   sudo dpkg -i  libcudnn8_   
   sudo dpkg -i  libcudnn8-dev
   ```    
## 二、下载pointRCNN，其基于pytorch1.0，修改适应ubuntu1804LTS  pytorch1.7    
1. 下载pointRCNN：参考pointRCNN中readme
    ```
    git clone --recursive https://github.com/sshaoshuai/PointRCNN.git
    pip install easydict tqdm tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```      
    <br/>  
2. pointRCNN源码解决pytorch版本不匹配问题   
   1）修改cuda支持问题
   ![](https://i.loli.net/2021/04/26/DaPCmJY3ez1QLv6.png)    
    ```
    grep AT_CHECK -rn .  ///> 在pointRCNN源码路径下搜索，并依次替换
    grep THCState_getCurrentStream(state) -rn .   ///> 在pointRCNN源码路径下搜索，并依次替换
    ```
    2）预训练模型(提供了car的预训练模型)直接受三个通道，因此需要将intensity排除
    ```
    __C.RPN.USE_INTENSITY = False   ///> pointRCNN/lib/config.py中
    ``` 
3. 安装一些库    
   ```
   sh build_and_install.sh
   ```
4. 评估预训练模型    
   1）将下载的KITTI数据集安装以下结构，放入pointRCNN源码源码目录中
   ![](https://i.loli.net/2021/04/26/6ZfIvmBwhsXFypu.png)
   <br/>   
   2）输出预测结果：data文件夹    
   运行以下语句将在/PointRCNN/output/rcnn/default/eval/epoch_no_number/val/final_result 路径下输出
   ```
   python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt PointRCNN.pth --batch_size 1 --eval_mode rcnn --set RPN.LOC_XZ_FINE False
   ```
   3）评估预测结果   
    &emsp;&emsp; a. write the CmakeLists.txt     
    ```
        ///  CmakeLists.txt
        cmake_minimum_required(VERSION 3.0)
        project(evaluate_detect_3d_offline)
        add_executable(evaluate_detect_3d_offline evaluate_object_3d_offline.cpp)
    ```      
     &emsp;&emsp; b. make generate  evaluate_detect_3d_offline      
     &emsp;&emsp; c. ./evaluate_detect_3d_offline gt_dir result_dir       
     &emsp;&emsp;&emsp;&emsp; 其中, gt_dir = Dataset/KITTI/data_object_label_2/training/label_2;    
     &emsp;&emsp;&emsp;&emsp;result_dir = /home/zc/MyFile/git/dataset/kitti_eval/pred;