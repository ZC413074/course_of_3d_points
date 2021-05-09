## FPFH思路
1. 采用ISS提取出点云的keypoints
2. 对每个keypoints计算FPFH
3. 在每个keypoint点处，利用公式计算出其与其近邻点之间的u,v,w，以及对应的alpha  phi theta
4. 分别建立alpha  phi theta的直方图，并直接连起来即可
## Test
1. 选择了机翼平面上的四个点，作为相似点
2. 分别计算其直方图，可以发现这四个点的直方图基本一致