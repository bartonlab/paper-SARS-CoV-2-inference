% this function converts a given codon to a unique codon number given by
% the mtx matrix

function thisCodon = num2codon(thisCodonNum)

mtx = [reshape(repmat([1 2 3 4], 16, 1), 64, 1) reshape(repmat([1 2 3 4], 4,4), 64, 1) repmat([1 2 3 4]', 16, 1)];
mtx = [mtx; 16 16 16];

thisCodon = mtx(thisCodonNum,:);