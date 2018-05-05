%%
%����Ƶ�еõ�ÿһ֡��ͼ��
fileName = 'your_video_path';     %����Ƶ�л�ȡÿһ֡��ͼ��
obj = VideoReader(fileName);
numFrames = obj.NumberOfFrames;% ֡������
for k = 1 : numFrames% ��ȡ����
    frame = read(obj,k);
    imshow(frame);%��ʾ֡
    imwrite(frame,strcat(num2str(k),'.jpg'),'jpg');% ����֡
end
%%
%����Ƶ�еõ���ÿһ֡ͼ����д���
for k = 1:numFrames
	filename = strcat('your_image_sequence_path\',int2str(k),'.jpg');
	A = imread(filename);     %��ȡͼ��
	pic=rgb2gray(A);       %�任
	[m,n] = size(pic);
	step_size = 8;         %������8
	D=[];
	patch = zeros(80,80);   %������С��80��80
	for t = 1:step_size:m-79          %���ղ�������
		 for l = 1:step_size:n-79
		   patch = A(t:t+79,l:l+79);  %���ջ�����Сȡ�ӿ�
		   C = reshape(patch,[80*80,1]); %��Ϊ������
		   D = [D C];        %��Ϊ������D��һ��
		 end
	end
	s = svd(double(D));    %�õ�D������ֵ����
	mu_k = s(2,1);           
	mu_bar = 0.05*s(4,1);
	lamda = 1/sqrt(max(m,n));
	eta = 0.9;               %�㷨����׼��
	B_k = gpuArray(zeros(size(D)));  %������ά���󣬵���ά��Ϊ��������¼�����ľ�����Ϣ���Ա��Ż��㷨ʵ��
	B_km1 = gpuArray(zeros(size(D)));  %ʹ��GPU���ټ���
	T_k = gpuArray(zeros(size(D)));
	T_km1 = gpuArray(zeros(size(D)));
	a_k = 1 ;
	a_km1 = 1 ;
	Iternum=1;
	G_k_B = gpuArray(single(D/2));
	converged = 0;
	threshold = 0.25;
	while Iternum<= 60   %�㷨����
		  Y_k_B = B_k + ((a_km1-1)/a_k)*(B_k-B_km1);
		  Y_k_T = T_k + ((a_km1-1)/a_k)*(T_k-T_km1);
		  G_k_B = Y_k_B - (Y_k_B + Y_k_T - single(D))/2;
		  [U,S,V] = svd(G_k_B,'econ');
		  B_kp1 = U*wthresh(S,'s',(mu_k)/2)*V';
		  G_k_T = Y_k_T - (Y_k_B + Y_k_T - single(D))/2;
		  T_kp1 = wthresh(G_k_T,'s',(lamda*mu_k)/2);
		  a_kp1 = (1+sqrt(4*a_k*a_k+1))/2;
		  %disp(num2str(a_kp1));    %��ʾ�����仯
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
	B = gather(B_kp1);    %��GPUarray�з��ص�double�;���
	T = gather(T_kp1);

	for i=1:529           %��patchͼ��patch��ԭͼ���е�λ��ͶӰ����ά������,����529��patch
		if (mod(i,23)==0)
			position = 23;
		else
			position = mod(i,23);
		end
		top_left_x = (i-position)/23*step_size;     %����ÿһ��patch��Ӧ��ԭͼ���е����Ͻ�����
		top_left_y = step_size*(position-1);
		T_sum(:,:,i)=zeros(m,n);
		B_sum(:,:,i)=zeros(m,n);
		T_sum((1:size(patch,1)) + top_left_x,(1:size(patch,1)) + top_left_y,i)=reshape(T(:,i),size(patch));
		B_sum((1:size(patch,2)) + top_left_x,(1:size(patch,2)) + top_left_y,i)=reshape(B(:,i),size(patch));
	end

	reconstruct_B = zeros(m,n);          %�ؽ�ͼ��
	reconstruct_T = zeros(m,n);
	for i = 1:m
		for j = 1:n
			temp2 = B_sum(i,j,:);
			position2 = find(temp2~=0);
			reconstruct_B(i,j) = median(reshape(temp2(find(temp2 ~= 0)),[1,length(position2)]));   %ʹ����ֵ������³����
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


	[row, col] = find( reconstruct_T > 0 );      %Ѱ���ؽ���Targetͼ���д���0�ĵ㣬�����龯������ֵ��Ϊ��
	if(isempty([row, col]) == 0)
		row = ceil(mean(row));                   %Ѱ��Target�����ĵ�����
		col = ceil(mean(col));
		pic(row-12:row-6 , col) = 255;           %�����
		pic(row+6:row+12 , col) = 255;
		pic(row , col-12:col-6) = 255;
		pic(row , col+6:col+12) = 255;
	end
	%imshow(pic);
	imwrite(pic,strcat('your_output_image_sequence_path\',int2str(k),'.jpg'));  %д������ļ�����
end
%%
%���������ͼ������������Ƶ
framesPath = 'your_output_image_sequence_path\'; %ͼ����������·����ͬʱҪ��֤ͼ���С��ͬ  
videoName = 'result.avi';%��ʾ��Ҫ��������Ƶ�ļ�������  
fps = 30; %֡��  
startFrame = 1; %����һ֡��ʼ  
endFrame = numFrames; %��һ֡����  
  
if(exist('videoName','file'))  
    delete videoName.avi  
end  
  
%������Ƶ�Ĳ����趨  
video_obj=VideoWriter(videoName);  %����һ��avi��Ƶ�ļ����󣬿�ʼʱ��Ϊ��  
video_obj.FrameRate=fps;  
  
open(video_obj); %Open file for writing video data  
%����ͼƬ  
for i=startFrame:endFrame  
    fileName=sprintf('%d',i);    %�����ļ������� �������ļ�����1.jpg 2.jpg ....  
    frames=imread([framesPath,fileName,'.jpg']);  
    writeVideo(video_obj,frames);  
end  
close(video_obj);% �رմ�����Ƶ 
