function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

for i=1:num_labels
    c= i==y;
    Y_out(:,i)=c;
end 

for i=1:m
    
    for j=1:hidden_layer_size
    hyp1(1,j)=sigmoid(Theta1(j,:)*transpose(X(i,:)));

    end
    hp1=[1 hyp1];
    for k=1:num_labels         
    hyp2(k,1)=sigmoid(Theta2(k,:)*transpose(hp1));
    end   
   f1(i,1)= sum(log(hyp2).*transpose(Y_out(i,:)));
   f2(i,1)= sum(log(1-hyp2).*(1-transpose(Y_out(i,:))));
end 

reg1=(lambda/(2*m))*sum(sum(Theta1(:,2:end).^2))
reg2=(lambda/(2*m))*sum(sum(Theta2(:,2:end).^2))

[f1r f1c]=size(f1)
[f2r f2c]=size(f2)
u1=sum(f1);
u2=sum(f2);
j1= -((1/m)*(u1+u2));
J=j1+reg1+reg2;

 delta1=zeros(size(Theta2));
 delta2=zeros(size(Theta1));

 
 
 
 % y(k) - the great trick - we need to recode the labels as vectors containing only values 0 or 1 (page 5 of ex4.pdf)
y_new = zeros(num_labels, m); % 10*5000
for i=1:m,
  y_new(y(i),i)=1;
end
Y_out=transpose(Y_out);
grad1 = zeros(size(Theta1));
grad2 = zeros(size(Theta2));
for t=1:m

    % Step 1
	a1 = X(t,:); % X already have a bias Line 44 (1*401)
    a1 = a1'; % (401*1)
	z2 = Theta1 * a1; % (25*401)*(401*1)
	a2 = sigmoid(z2); % (25*1)
    
    a2 = [1 ; a2]; % adding a bias (26*1)
	z3 = Theta2 * a2; % (10*26)*(26*1)
	a3 = sigmoid(z3); % final activation layer a3 == h(theta) (10*1)
    
    % Step 2
	delta_3 = a3 - Y_out(:,t); % (10*1)
	
    z2=[1; z2]; % bias (26*1)
    % Step 3
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % ((26*10)*(10*1))=(26*1)

    % Step 4
	delta_2 = delta_2(2:end); % skipping sigma2(0) (25*1)

	grad2 = grad2 + delta_3 * a2'; % (10*1)*(1*26)
	grad1 = grad1 + delta_2 * a1'; % (25*1)*(1*401)
    
end;

% Step 5
grad2 = (1/m) * grad2; % (10*26)
grad1 = (1/m) * grad1; % (25*401)

%regularization 

% Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m; % for j = 0
% 
grad1(:, 2:end) = grad1(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
% 
% Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m; % for j = 0
% 
grad2(:, 2:end) = grad2(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

% Unroll gradients
grad = [grad1(:) ; grad2(:)];



% 

end
