function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;     
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

m=length(X)
n=size(X)
X1=[ones(length(X),1) X];

for i=1:hidden_layer_size
z1(:,i)=sigmoid(X1*transpose(Theta1(i,:)));
end 
z1=[ones(length(X),1) z1]
for i=1:num_labels
z2(:,i)=sigmoid(z1*transpose(Theta2(i,:)));
end 
[M,I]=max(z2, [], 2);


p=I;

% =========================================================================


end
