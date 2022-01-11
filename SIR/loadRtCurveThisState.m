

if(RtSet == 3)
    citySetStr = [citySetStr '_Profile' num2str(RtSet)];
    rtThisState = [3.875004827
    3.860465652
    3.859493744
    3.848117907
    3.832177525
    3.804057253
    3.767029756
    3.714552637
    3.641352159
    3.544856644
    3.425039807
    3.279538608
    3.104868552
    2.91290213
    2.708980685
    2.50688316
    2.29868999
    2.105353127
    1.921536464
    1.755528892
    1.608596659
    1.480670803
    1.366900141
    1.270450346
    1.184661328
    1.110599498
    1.046920384
    0.991658887
    0.947315691
    0.909065253
    0.874781758
    0.843731067
    0.817815883
    0.797780664
    0.779431546
    0.764357372
    0.749839456
    0.738505963
    0.726487584
    0.718380989
    0.712113512
    0.704943939
    0.699748974
    0.694028376
    0.690359994
    0.685893792
    0.683126468
    0.68119152
    0.679303743
    0.679517394
    0.678007989
    0.678051096
    0.677276427
    0.676839166
    0.677548242
    0.6780164
    0.68004485
    0.680766587
    0.680470639
    0.680629036
    0.681824004
    0.68251908
    0.684058192
    0.685657898
    0.687010669
    0.687672026
    0.689290299
    0.690033488
    0.691743106
    0.693703353
    0.694967323
    0.695960936
    0.697598357
    0.69984764
    0.701166899
    0.702668861
    0.705850394
    0.709038831
    0.711763531
    0.715677973
    0.721253519
    0.726944843
    0.733148731
    0.739609021
    0.745287852
    0.752763323
    0.760677265
    0.769619449
    0.780010381
    0.78961484
    0.802290126
    0.814321957
    0.829551242
    0.84113416
    0.855501745
    0.867865309
    0.88497347
    0.89913073
    0.912608218
    0.927214259
    0.941112921
    0.955983055
    0.968405584
    0.97959283
    0.99192985
    1.001771163
    1.011632052
    1.018789732
    1.02515417
    1.030414576
    1.035601801
    1.038778231
    1.041971062
    1.044720105
    1.044350033
    1.043933613
    1.044271564
    1.043267292
    1.042179752
    1.037119501
    1.034390449
    1.03201018
    1.027840245
    1.02598758
    1.020098202
    1.017641963
    1.015125191
    1.010763355
    1.006351375
    1.002674835
    0.994553351
    0.990494327
    0.984813004
    0.980554037
    0.975376713
    0.969946738
    0.964305862
    0.961369632
    0.957812306
    0.954722664
    0.950364944
    0.948282925
    0.945965766
    0.943375109
    0.94055648
    0.937694616
    0.937558767
    0.936171098
    0.935274577
    0.934190634
    0.933773166
    0.933022581
    0.933761932
    0.932615928
    0.934000708
    0.935843979
    0.938196387
    0.939489989
    0.941853277
    0.941443251
    0.942090082
    0.94392908
    0.944047148
    0.947040854
    0.949771004
    0.951096282
    0.952260432
    0.952706957
    0.954540373
    0.956761194
    0.958544174
    0.960044068
    0.960218515
    0.96186546
    0.962040491
    0.961186193
    0.964532891
    0.964274364
    0.965992454
    0.966009668
    0.967654387
    0.969396211
    0.970894104
    0.972104802
    0.971963143];
elseif(RtSet == 1)
    citySetStr = [citySetStr '_Profile' num2str(RtSet)];
    x = 1:1:tmax;
    y = 1*(2 - 1.5./(1+exp(-.12*(x-40))));
    rtThisState = y;
elseif(RtSet == 2)
    citySetStr = [citySetStr '_Profile' num2str(RtSet)];
    v = 1:1:tmax;
    z = exp(-v/100) + 0.5 + 0.25*(sin(2*pi*4*v/100));
    % ggt = 25;
    % z = [z(1:ggt) ones(1, tmax-ggt)];
    rtThisState = z;
elseif(RtSet == 4)
    citySetStr = [citySetStr '_Profile' num2str(RtSet)];
    v = 0.01:0.01:1.85;
    temp1 = [3:-0.01:2.8 2.8:0.1:0.8 0.8:0.1:1 ones(1,10) 1:0.2:1.6 1.6:0.2:3 3:-0.01:2.8 2.8:0.1:0.8 0.8:0.1:1 ones(1,10)];
    temp2 = repmat(temp1, 1, 3);
    z = temp2(1:length(v));
    % ggt = 25;
    % z = [z(1:ggt) ones(1, tmax-ggt)];
    rtThisState = z;
elseif(RtSet == 5)
    citySetStr = [citySetStr '_Profile' num2str(RtSet)];
    rtThisState = 1*ones(1, tmax);
elseif(RtSet == 6)
    citySetStr = [citySetStr '_Profile' num2str(RtSet)];
    rtThisState = 2*ones(1, tmax);
elseif(RtSet == 7)
    citySetStr = [citySetStr '_Profile' num2str(RtSet)];
    rtThisState = 4*ones(1, tmax);end
%rtThisState(rtThisState > 1.8) = rtThisState(rtThisState > 1.8)/2;
%rtThisState(rtThisState > -1) = 4;

% rtThisState = rtThisState + 0.75;
% rtThisStateTemp = [1.5*ones(1,16) 1.2 0.8*ones(1,30) 0.9 0.95*ones(1,120+200)];
% t_temp =          [        0:1:15   22      26:1:55 62 66:1:(tmax+200)];
% rtThisState = interp1(t_temp, rtThisStateTemp, 0:1:(184+200));

thisStateMeanR0 = rtThisState(1);

rtThisStateNorm = rtThisState/rtThisState(1);