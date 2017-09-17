function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    delta = (1/m)*sum(X.*repmat((X*theta - y), 1, size(X,2)));
    theta = (theta' - (alpha * delta))';

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
% == To test pass the following argument
% X = m*n matrix  // example (1 2; 3 4; 5 6) 3*2 size
% y = m*1 column vector // example (1; 2; 3) 3*1 size
% theta = n*1 column vector // example ([1;2]) 2*1 size
% alpha = singular value
