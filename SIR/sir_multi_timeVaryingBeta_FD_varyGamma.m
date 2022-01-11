
% for simplifying calculations, the /N part is included in beta, and not
% with I

% need to send in sperate N as in other function N is included in beta
function f = sir_multi_timeVaryingBeta_FD_varyGamma(t,y, beta_t, betaTimeVarying, newInfected_Base, param, N, betaForConstInfcFileName)
% betaTimeVarying is input but not used...only used for testing code
% purpose

% y(1) = S, y(2) = I, y(3) = R
dim = length(y)/3;

mud = param(1:dim); 
gamma = param((1*dim+1):2*dim);
mub = param((2*dim+1):3*dim);
% sigma_ij assumed to be 1

betaInterpMax = zeros(dim,1);
newInfected_Interp_Base = zeros(dim,1);
for i = 1:dim
    %betaInterp(i) = interp1(beta_t, betaTimeVarying(i,:), t); % Interpolate the data set (beta_t, beta) at times t
    betaInterpMax(i) = interp1(beta_t, betaTimeVarying(i,:), t); % Interpolate the data set (beta_t, beta) at times t
    newInfected_Interp_Base(i) = interp1(beta_t, newInfected_Base(i,:), t);
end

S = y(1:dim);
I = y((dim+1):2*dim);
R = y((2*dim+1):3*dim);
Sdot = zeros(dim, 1);
Idot = zeros(dim, 1);
Rdot = zeros(dim, 1);


betaInterp = betaInterpMax;
condition = sum(betaInterp.*I)*S(1) ~= sum(newInfected_Interp_Base);

if(condition == true)
    tempFac = sum(betaInterp.*I)*S(1)/sum(newInfected_Interp_Base);
    betaInterp = betaInterp/tempFac;
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