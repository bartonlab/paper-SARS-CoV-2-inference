
% for simplifying calculations, the /N part is included in beta, and not
% with I
function f = sir_multi_timeVaryingBeta_FD_constInfc(t,y, beta_t, betaTimeVarying, param, constPopSize, N, betaForConstInfcFileName)

% y(1) = S, y(2) = I, y(3) = R
dim = length(y)/3;

mud = param(1:dim); 
gamma = param((1*dim+1):2*dim);
mub = param((2*dim+1):3*dim);
% sigma_ij assumed to be 1

betaInterpMax = zeros(dim,1);
for i = 1:dim
    betaInterpMax(i) = interp1(beta_t, betaTimeVarying(i,:), t); % Interpolate the data set (beta_t, beta) at times t
end

S = y(1:dim);
I = y((dim+1):2*dim);
R = y((2*dim+1):3*dim);
Sdot = zeros(dim, 1);
Idot = zeros(dim, 1);
Rdot = zeros(dim, 1);

betaInterp = betaInterpMax;
condition = sum(betaInterp.*I)*S(1) >= constPopSize;


if(condition == true )
    tempFac = sum(betaInterp.*I)*S(1)/constPopSize;
%     S(1) 
%     [I' sum(I)]
%     [sum(betaInterp.*I)*S(1) constPopSize]
%     tempFac
    
    
    betaInterp = betaInterp/tempFac;
    
%     betaInterp'
%     t
%     pause
else
    betaInterp = betaInterpMax;
end

dlmwrite(betaForConstInfcFileName, [t N*betaInterp'], '-append')


for i = 1:length(S)
    temp = 0;
    
    for j = 1:length(S)
        temp = temp + betaInterp(j)*S(i)*I(j);
    end
    Sdot(i) =  mub(i) - mud(i)*S(i) - temp;
    Idot(i) = betaInterp(i)*S(i)*I(i) - mud(i)*I(i) - gamma(i)*I(i);
    Rdot(i) = gamma(i)*I(i) - mud(i)*R(i);
end

f = [Sdot; Idot; Rdot];