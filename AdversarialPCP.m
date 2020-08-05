function [S,D,obj]=AdversarialPCP(W,P,T,para)
%%%%%%% function readme %%%%%%%%
% the function solves the Adversarial pairwise constraints propagation probelm,
% i.e, min |S\odotD|+alpha*trace(D*L*D')+beta*trace(S*L*S')+gamma*norm(P.*(D-C))+gamma*norm(P.*(S-M))
% the first term is the Adversarial term
% the second term and the third term achieve pairwise constraitn  propagation propagation,
% and the last two terms introduce the initital pairwise constraint information 

% W: local similarity matrix for 
% A: diagonal degree matrix for W
% L: the Laplacian matrix for W 
% P the position matrix of the supervisory information (both must-link and cannot-link)
% T: the cannot-lilnk information matrix
%%%%%%% parameter setting
A=diag(sum(W,2));
L=A-W;
Z1=P-T; % initial S (must-link matrix)
Z2=T; % initial D (cannot-link matrix)

alpha=para.alpha;
beta=para.beta;
gamma=para.gamma;
maxiter=para.maxiter;

S=rand(size(W));
D=rand(size(W));

obj(1)=sum(sum(S.*D))+alpha*trace(D*L*D')+beta*trace(S*L*S')...
    +gamma*norm(P.*(D-Z2))+gamma*norm(P.*(S-Z1));

for iter=1:maxiter
    
% update S and D
    S=S.*((2*gamma*(P.*Z1)+2*beta*S*W)./(D+2*beta*S*A+2*gamma*(P.*S))+eps);
    D=D.*((2*gamma*(P.*Z2)+2*alpha*D*W)./(S+2*alpha*D*A+2*gamma*(P.*D))+eps);
    obj(iter+1)=sum(sum(S.*D))+alpha*trace(D*L*D')+beta*trace(S*L*S')...
        +gamma*norm(P.*(D-Z2))+gamma*norm(P.*(S-Z1));

    disp(['the ',num2str(iter),' iteration. obj value: ',num2str(obj(iter+1))]);

    if max(max(abs(obj(iter)-obj(iter+1))))<10^-3 
        break;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S=(S+S')/2;
    D=(D+D')/2;
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end



