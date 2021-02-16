clear all
close all
clc
addpath('E:\RegistrarBKP\Registrar\Evaluation\NeuronTracerV20')
user = ['AS';'RG';'JC'];
numuser = 3;

for num = 1:6
    % Create Scenarios
    parameter_file_path='C:\Users\Seyed\Documents\DatasetTests\AutomatedTracing\AutomatedTracing\NeuronTracerV20\Parameter Files\Parameters_L6.txt';
    Parameters=Read_Parameter_File(parameter_file_path);
    G = load(['E:\AutomatedTracing\Data\Traces\L1\',num2str(num),'_L6_',user(numuser,:),'.mat']);
    AM = G.AM;
    r = G.r;
    R = G.R;
    Orig = double(G.Original)/255;
    [AMlbl,r,R,ClustersStr]=GenerateClustersStrW_new1(G.Original,AM,r,R,Parameters);
    
%     figure
%     imshow(max(Orig,[],3))
%     hold on
%     PlotAM_1(AM,r)
%     hold on
%     Cluster_r =  ClustersStr(10).end_point_r;
%     Cluster_n =  ClustersStr(10).end_point_n;
%     Cluster_center = round([sum(Cluster_r(:,1))/length(Cluster_r(:,1)), sum(Cluster_r(:,2))/length(Cluster_r(:,2)), sum(Cluster_r(:,3))/length(Cluster_r(:,3))]);
%     plot3(Cluster_r(:,2),Cluster_r(:,1),Cluster_r(:,3),'r*')
    
%     for i=1:size(Cluster_n,1)
%         line([Cluster_r(i,2),Cluster_r(i,2)+Cluster_n(i,2)],[Cluster_r(i,1),Cluster_r(i,1)+Cluster_n(i,1)],[Cluster_r(i,3),Cluster_r(i,3)+Cluster_n(i,3)])
%     end
    save(['E:\AutomatedTracing\Data\Traces\L1\',num2str(num),'_L6_',user(numuser,:),'_withALLClusters1_NewFeatures.mat']);
end

pad = 6;
TCounter = 1;

 for n = 1:6
    D1 = load(['E:\AutomatedTracing\Data\Traces\L1_org\',num2str(n),'_L6_',user(numuser,:),'_withALLClusters1.mat']);
    D2 = load(['E:\AutomatedTracing\Data\Traces\L1\',num2str(n),'_L6_',user(numuser,:),'_withALLClusters1_NewFeatures.mat']);
%     D1 = D2;
    ClustersStr = D1.ClustersStr;
    ClustersStr_NewFeatures = D2.ClustersStr;
    
    for numCluster = 1: size(ClustersStr,2)
        if size(ClustersStr(numCluster).scenarios,3) > 100
            ClustersStr(numCluster).scenarios = ClustersStr(numCluster).scenarios(:,:,1:100);
            ClustersStr(numCluster).cost_components = ClustersStr(numCluster).cost_components(:,1:100);
            ClustersStr(numCluster).best_merger = ClustersStr(numCluster).best_merger(:,1:100);
            ClustersStr(numCluster).alpha = ClustersStr(numCluster).alpha(:,1:100);
            
            ClustersStr_NewFeatures(numCluster).scenarios = ClustersStr_NewFeatures(numCluster).scenarios(:,:,1:100);
            ClustersStr_NewFeatures(numCluster).cost_components = ClustersStr_NewFeatures(numCluster).cost_components(:,1:100);
            ClustersStr_NewFeatures(numCluster).best_merger = ClustersStr_NewFeatures(numCluster).best_merger(:,1:100);
            ClustersStr_NewFeatures(numCluster).alpha = ClustersStr_NewFeatures(numCluster).alpha(:,1:100);
        end
    end
    
    
    GG = load(['E:\AutomatedTracing\Data\Traces\L1_org\',num2str(n),'_L6_',user(numuser,:),'.mat']);
    Orig = double(GG.Original)/255;
    Orig = padarray(Orig,[pad,pad,pad]);
    
    for numCluster = 1:size(ClustersStr,2)
        scenarios = ClustersStr(numCluster).scenarios;
        cost_components = ClustersStr(numCluster).cost_components;
        best_merger =  ClustersStr(numCluster).best_merger;
        Cluster_r =  ClustersStr(numCluster).end_point_r+pad;
        Cluster_center = round([sum(Cluster_r(:,1))/length(Cluster_r(:,1)), sum(Cluster_r(:,2))/length(Cluster_r(:,2)), sum(Cluster_r(:,3))/length(Cluster_r(:,3))]);
        end_point_n = ClustersStr_NewFeatures(numCluster).end_point_n;
        
        local_Cluster_r_augm = [];
        end_point_n_augm = [];
        ClusterIM_augm = [];
        
        Start = Cluster_center - pad;
        End = Cluster_center + pad;
        local_Cluster_r = Cluster_r - Start + 1;
        
        ClusterIM = Orig(Start(1):End(1),Start(2):End(2),Start(3):End(3));
        
        for NumMerger = 1:size(best_merger,2)
            TData(TCounter).Label = best_merger(NumMerger) ==1;
            TData(TCounter).IM = ClusterIM;
            TData(TCounter).IMnum = n;
            TData(TCounter).Scenario = scenarios(:,:,NumMerger);
            % NEW FEATURES
            TData(TCounter).NewFeatures = Generate_New_Features(local_Cluster_r,end_point_n,scenarios(:,:,NumMerger));
%             TData(TCounter).NewFeatures = Generate_New_Features(local_Cluster_r,end_point_n,scenarios(:,:,NumMerger));
            
            TCounter = TCounter + 1;
        end
    end
end

for i = 1:size(TData,2)
    Labels{i} = TData(i).Label;
    IMs{i} = TData(i).IM;
    IMnum{i} = TData(i).IMnum;
    NewFeatures{i} = TData(i).NewFeatures;
    Scenarios{i} = TData(i).Scenario;
end
path = ['E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\IMonce_100_scen_NEW_Inv_FEATURES_User=',user(numuser,:),'.mat'];

% % path = ['E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\All_scenarios_label_14Features_Image_Trace_new_withIMnum_with_local_XYZ_NewFeatures_Fixedlocal_Augmented_IM=',num2str(n),'.mat'];
save(path,'IMs','Scenarios','Labels','IMnum','NewFeatures');

