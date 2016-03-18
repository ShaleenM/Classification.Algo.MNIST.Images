%% Get Data

train_image_file = '/Users/Mathurs/Google Drive/UB/Courses(UB)/CSE574- Machine Learning/Project/Project3/data/train-images-idx3-ubyte';
train_label_file = '/Users/Mathurs/Google Drive/UB/Courses(UB)/CSE574- Machine Learning/Project/Project3/data/train-labels-idx1-ubyte';
x_train = loadMNISTImages(train_image_file);
t_train = loadMNISTLabels(train_label_file);

test_image_file = '/Users/Mathurs/Google Drive/UB/Courses(UB)/CSE574- Machine Learning/Project/Project3/data/t10k-images-idx3-ubyte';
test_label_file = '/Users/Mathurs/Google Drive/UB/Courses(UB)/CSE574- Machine Learning/Project/Project3/data/t10k-labels-idx1-ubyte';
x_test = loadMNISTImages(test_image_file);
t_test = loadMNISTLabels(test_label_file);

clear train_image_file train_label_file test_image_file test_label_file
save('data/data_MNIST.mat','x_train','t_train','x_test','t_test')


%% Logistic Regression

clear all
load ('data/data_MNIST.mat');
N = 60000;
N2 = 10000;
D = 784;
K = 10;

%% Training 

eta = 0.001;            % Fixed

blr = 0.1*ones(1,K);
x_temp = ones(1,N);
x_train = [x_train ;  x_temp];
Wlr = zeros(D,K);
Wlr = [Wlr; blr];

t_train_k = zeros(N,K);
for n = 1:N
    t_train_k(n,t_train(n)+1)=1;
end
 error_prev = 0;
for iteration = 1 : 5         
    for tau = 1: N
        a = zeros(1,K);
        y = zeros(1,K);
        err = zeros(D+1,K);
        for k= 1:K
            a(k) = (Wlr(:,k)'* x_train(:,tau)) ;
        end
        a_max = max(a);
        den_y = sum(exp(a./a_max)); 
        y = exp(a/a_max)/(den_y);
        y_max = max(y);
        for k = 1:K 
            if y(k) == y_max
                y(k) = 0.9999;
            else
                y(k) = 0.0001;
            end
            err(:,k) = ((y(k)-t_train_k(tau,k))*x_train(:,tau)')';
            Wlr(:,k) = Wlr(:,k) - (eta .* err(:,k));
        end
        
            error = -sum(t_train_k(tau,:) .* log(y))./K;
            if abs(error) < abs(error_prev) && eta > 0.00001        
                eta = eta*0.9995;
            end
        error_prev = error;
    end  
end

%% Testing Error

N2=10000;
right_count=0;
wrong_count=0;
 
x_temp = ones(1,N2);
x_test = [x_test ;  x_temp];
for tau = 1: N2
        a = zeros(1,K);
        y = zeros(1,K);
        for k= 1:K
            a(k) = (Wlr(:,k)'* x_test(:,tau)) ;
        end
        a_max = max(a);
        den_y = sum(exp(a./a_max));
        for k = 1:K 
            y(k) = exp(a(k)/a_max)/(den_y);
        end
      [val idx]=max(y);
	
	if((idx-1)==t_test(tau,1))
		right_count=right_count+1;
	else
		wrong_count=wrong_count+1;
    end
end
    fprintf('\nTesting Error :\n');
    fprintf('Misclassification Rate : %f \n',(wrong_count*100/N2));
blr = Wlr(end,:);
Wlr = Wlr (1:end-1,:);
x_train = x_train (1:end-1,:);
save('proj3.mat', 'Wlr' ,'blr')

%% Neural Network

clear all
load ('data/data_MNIST.mat');
N = 60000;
D = 784;
K = 10;
J = 600;
blk = 50;
h = 'sigmoid';

%% Functions

sigmoid = @(a) 1.0./(1.0 + exp(-a));

%% Training

eta =  blk*0.0001 ;
Wnn1 = randn(D,J);
Wnn2 = randn(J,K);
bnn1 =  0.8*ones(1,J);
bnn2 =  0.8*ones(1,K);

x_temp = ones(1,N);
x_train = [x_train ;  x_temp];
Wnn1 = [Wnn1; bnn1]; 

t_train_k = zeros(N,K);         
for n = 1:N
    t_train_k(n,t_train(n)+1)=1;          
end
for iteration = 1:1
    fprintf('%d\n',iteration);
    for tau = blk:blk:N
        x_blk = x_train(:,tau-blk+1:tau);
        y = zeros(K,blk);
        aj = (Wnn1'* x_blk); 
        z = sigmoid(aj);
        ak = [Wnn2;bnn2]'* [z ; ones(1,blk)];
        den_y = sum(exp(ak));
        for i= 1:blk
            y(:,i) = exp(ak(:,i))./(sum(exp(ak(:,i))));
            y_max = max(y(:,i));
            for k = 1:K
                if y(k,i) == y_max
                        y(k,i) = 0.9999;
                    else
                        y(k,i) = 0.0001;
                end
            end
        end
        del_k = y' - t_train_k(tau-blk+1:tau,:);
        err1 = Wnn2*del_k';
        del_j = (sigmoid(z).*(ones(J,blk) - sigmoid(z))).*(err1) ;
        Wnn1 = Wnn1 - (eta/blk).*(x_blk*del_j');
        Wnn2 = Wnn2 - ((eta/blk).*(z*del_k));
    end
end
save('proj3.mat', 'Wnn1' ,'Wnn2','bnn1','bnn2','h', '-append')

bnn1 = Wnn1(end,:);
Wnn1 = Wnn1 (1:end-1,:);
%% Testing

N1 = 10000;
right_count=0;
wrong_count=0;

for tau = 1:N1
    aj = zeros(1,J);
    ak = zeros(1,K);
    z = zeros(1,J);
    y = zeros(1,K);
    
    for j= 1:J
        aj(j) = (Wnn1(:,j)'* x_test(:,tau)) + bnn1(j); 
        z(j) = sigmoid(aj(j));                  %sigmoid  % activation of hidden layer functionx`
    end
    
    for k = 1:K 
        ak(k) = (Wnn2(:,k)'* z') + bnn2(k);
    end
    
    den_y = sum(exp(ak));
    for k = 1:K 
        y(k) = exp(ak(k))./(den_y);
    end

        [val idx]=max(y);
	
	if((idx-1)==t_test(tau,1))
		right_count=right_count+1;  
	else
		wrong_count=wrong_count+1;
    end
end
    fprintf('Itertion : %d \t',iteration);
    fprintf('Right : %d \t',right_count);
    fprintf('Wrong : %d \n',wrong_count);


save('proj3.mat', 'Wnn1' ,'Wnn2','bnn1','bnn2','h', '-append')