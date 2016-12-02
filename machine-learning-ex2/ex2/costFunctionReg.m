function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%num=size(theta,1)
L=theta
L(1)=0
J=(1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))+sum(L.^2)*lambda*(2*m)^-1
%grad(2:num)=(1/m)*X(:,2:num)'*(sigmoid(X(:,2:num)*theta(2:num))-y)+lambda*theta(2:num)*m^-1
%grad(1)=(1/m)*X(:,1)'*(sigmoid(X(:,1)*theta(1))-y)
grad=(1/m)*X'*(sigmoid(X*theta)-y)+lambda*L*(m^-1)



% =============================================================

end
