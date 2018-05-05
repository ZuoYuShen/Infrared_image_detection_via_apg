%%
%从视频中得到每一帧的图像
fileName = 'your_video_path';     %从视频中获取每一帧的图像
obj = VideoReader(fileName);
numFrames = obj.NumberOfFrames;% 帧的总数
for k = 1 : numFrames% 读取数据
    frame = read(obj,k);
    imshow(frame);%显示帧
    imwrite(frame,strcat(num2str(k),'.jpg'),'jpg');% 保存帧
end
%%
%从视频中得到的每一帧图像进行处理
for k = 1:numFrames
	filename = strcat('your_image_sequence_path\',int2str(k),'.jpg');
	A = imread(filename);     %读取图像
	pic=rgb2gray(A);       %变换
	[m,n] = size(pic);
	step_size = 8;         %步长：8
	D=[];
	patch = zeros(80,80);   %滑窗大小：80×80
	for t = 1:step_size:m-79          %按照步长设置
		 for l = 1:step_size:n-79
		   patch = A(t:t+79,l:l+79);  %按照滑窗大小取子块
		   C = reshape(patch,[80*80,1]); %变为列向量
		   D = [D C];        %作为矩阵中D的一列
		 end
	end
	s = svd(double(D));    %得到D的奇异值矩阵
	mu_k = s(2,1);           
	mu_bar = 0.05*s(4,1);
	lamda = 1/sqrt(max(m,n));
	eta = 0.9;               %算法数据准备
	B_k = gpuArray(zeros(size(D)));  %构建三维矩阵，第三维作为索引，记录迭代的矩阵信息，以便优化算法实现
	B_km1 = gpuArray(zeros(size(D)));  %使用GPU加速计算
	T_k = gpuArray(zeros(size(D)));
	T_km1 = gpuArray(zeros(size(D)));
	a_k = 1 ;
	a_km1 = 1 ;
	Iternum=1;
	G_k_B = gpuArray(single(D/2));
	converged = 0;
	threshold = 0.25;
	while Iternum<= 60   %算法迭代
		  Y_k_B = B_k + ((a_km1-1)/a_k)*(B_k-B_km1);
		  Y_k_T = T_k + ((a_km1-1)/a_k)*(T_k-T_km1);
		  G_k_B = Y_k_B - (Y_k_B + Y_k_T - single(D))/2;
		  [U,S,V] = svd(G_k_B,'econ');
		  B_kp1 = U*wthresh(S,'s',(mu_k)/2)*V';
		  G_k_T = Y_k_T - (Y_k_B + Y_k_T - single(D))/2;
		  T_kp1 = wthresh(G_k_T,'s',(lamda*mu_k)/2);
		  a_kp1 = (1+sqrt(4*a_k*a_k+1))/2;
		  %disp(num2str(a_kp1));    %显示参数变化
		  mu_kp1 = max(eta*mu_k, mu_bar);
		  %disp(num2str(mu_kp1));
		  
		  E_B = B_kp1 - B_k;
		  E_T = T_kp1 - T_k;
		  
		  stop = norm([E_B,E_T],'fro');
		  %disp(num2str(stop));
		  if stop <= threshold
			 converged = 1;
		  end
		  
		  B_km1 = B_k;
		  T_km1 = T_k;
		  B_k = B_kp1;
		  T_k = T_kp1;
		  a_km1 = a_k;
		  a_k = a_kp1;
		  mu_k = mu_kp1;
		  
		  Iternum = Iternum + 1;
		  %disp(num2str(Iternum));
	end
	B = gather(B_kp1);    %从GPUarray中返回到double型矩阵
	T = gather(T_kp1);

	for i=1:529           %将patch图像按patch在原图像中的位置投影到三维矩阵中,共有529个patch
		if (mod(i,23)==0)
			position = 23;
		else
			position = mod(i,23);
		end
		top_left_x = (i-position)/23*step_size;     %计算每一个patch对应在原图像中的左上角坐标
		top_left_y = step_size*(position-1);
		T_sum(:,:,i)=zeros(m,n);
		B_sum(:,:,i)=zeros(m,n);
		T_sum((1:size(patch,1)) + top_left_x,(1:size(patch,1)) + top_left_y,i)=reshape(T(:,i),size(patch));
		B_sum((1:size(patch,2)) + top_left_x,(1:size(patch,2)) + top_left_y,i)=reshape(B(:,i),size(patch));
	end

	reconstruct_B = zeros(m,n);          %重建图像
	reconstruct_T = zeros(m,n);
	for i = 1:m
		for j = 1:n
			temp2 = B_sum(i,j,:);
			position2 = find(temp2~=0);
			reconstruct_B(i,j) = median(reshape(temp2(find(temp2 ~= 0)),[1,length(position2)]));   %使用中值更具有鲁棒性
		   if(isempty(find(T_sum(i,j,:) ~= 0, 1)))
			temp1 = 0;
			reconstruct_T(i,j) = 0;
		   else
			temp1 = T_sum(i,j,:);
			position1 = find(temp1~=0);
			reconstruct_T(i,j) = median(reshape(temp1(find(temp1 ~= 0)),[1,length(position1)]));
		   end
		end
	end


	[row, col] = find( reconstruct_T > 0 );      %寻找重建的Target图像中大于0的点，发现虚警的像素值都为负
	if(isempty([row, col]) == 0)
		row = ceil(mean(row));                   %寻找Target的中心点坐标
		col = ceil(mean(col));
		pic(row-12:row-6 , col) = 255;           %做标记
		pic(row+6:row+12 , col) = 255;
		pic(row , col-12:col-6) = 255;
		pic(row , col+6:col+12) = 255;
	end
	%imshow(pic);
	imwrite(pic,strcat('your_output_image_sequence_path\',int2str(k),'.jpg'));  %写入输出文件夹中
end
%%
%根据输出的图像序列生成视频
framesPath = 'your_output_image_sequence_path\'; %图像序列所在路径，同时要保证图像大小相同  
videoName = 'result.avi';%表示将要创建的视频文件的名字  
fps = 30; %帧率  
startFrame = 1; %从哪一帧开始  
endFrame = numFrames; %哪一帧结束  
  
if(exist('videoName','file'))  
    delete videoName.avi  
end  
  
%生成视频的参数设定  
video_obj=VideoWriter(videoName);  %创建一个avi视频文件对象，开始时其为空  
video_obj.FrameRate=fps;  
  
open(video_obj); %Open file for writing video data  
%读入图片  
for i=startFrame:endFrame  
    fileName=sprintf('%d',i);    %根据文件名而定 我这里文件名是1.jpg 2.jpg ....  
    frames=imread([framesPath,fileName,'.jpg']);  
    writeVideo(video_obj,frames);  
end  
close(video_obj);% 关闭创建视频 
