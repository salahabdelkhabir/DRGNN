import axios from 'axios';
import { IAttentionTree, IEdgeTypes, IPath, IState } from 'types';
import { SERVER_URL, DATA_URL } from 'Const';

const axiosInstance = axios.create({
  baseURL: `${SERVER_URL}/`,
  // timeout: 1000,
  withCredentials: false,
  headers: {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,PATCH,OPTIONS',
  },
});

const requestNodeTypes = async (): Promise<string[]> => {
  const url = `./${DATA_URL}/node_types.json`;
  let response = await axiosInstance.get(url);
  return response.data;
};

const requestEdgeTypes = async (): Promise<IEdgeTypes> => {
  const url = `./${DATA_URL}/edge_types.json`;
  let response = await axiosInstance.get(url);
  return response.data;
};

const requestNodeNameDict = async () => {
  const url = `./${DATA_URL}/node_name_dict.json`;
  let response = await axiosInstance.get(url);
  return response.data;
};

const requestAttention = async (diseaseID: string, drugID: string) => {
  const url = `./api/attention?disease=${diseaseID}&drug=${drugID}`;
  let response = await axiosInstance.get(url);
  return response.data;
};

const requestAttentionPair = async (
  diseaseID: string,
  drugID: string
): Promise<{ attention: { [k: string]: IAttentionTree }; paths: IPath[] }> => {
  console.log(`Requesting attention pair for disease: ${diseaseID}, drug: ${drugID}`);
  try {
    const url = `./api/attention_pair?disease=${diseaseID}&drug=${drugID}`;
    let response = await axiosInstance.get(url);
    console.log('Attention pair response:', response.status);
    
    // Validate response data
    if (!response.data || !response.data.attention) {
      console.error('Invalid attention pair response:', response.data);
      // Return empty data
      return { attention: {}, paths: [] };
    }
    
    return response.data;
  } catch (error) {
    console.error('Error in requestAttentionPair:', error);
    // Return empty data on error
    return { attention: {}, paths: [] };
  }
};
const requestDiseaseOptions = async () => {
  const urlRanking = `./${DATA_URL}/disease_options.json`;
  console.log("Requesting disease options from:", urlRanking);
  
  try {
    let res = await axiosInstance.get(urlRanking);
    console.log("Disease options data length:", res.data.length);
    return res.data;
  } catch (error) {
    console.error("Error loading disease options:", error);
    return [];
  }
};

const requestDrugPredictions = async (diseaseID: string) => {
  console.log(`Frontend - requestDrugPredictions - Requesting predictions for disease: ${diseaseID}`);
  
  if (!diseaseID) {
    console.warn('Frontend - requestDrugPredictions - Empty diseaseID provided');
    return [];
  }
  
  try {
    const url = `./api/drug_predictions?disease_id=${diseaseID}`;
    console.log(`Frontend - requestDrugPredictions - Making request to: ${url}`);
    
    const response = await axiosInstance.get(url);
    console.log(`Frontend - requestDrugPredictions - Response status: ${response.status}`);
    console.log(`Frontend - requestDrugPredictions - Response data length: ${response.data ? response.data.length : 0}`);
    
    if (response.data && Array.isArray(response.data) && response.data.length === 0) {
      console.warn('Frontend - requestDrugPredictions - Empty predictions returned');
    }
    
    return response.data || [];
  } catch (error) {
    console.error('Frontend - requestDrugPredictions - Request failed:', error);
    return [];
  }
};

const requestEmbedding = async () => {
  const url = `./${DATA_URL}/drug_tsne.json`;
  const response = await axiosInstance.get(url);
  return response.data;
};

export {
  requestNodeTypes,
  requestEdgeTypes,
  requestAttention,
  requestNodeNameDict,
  requestDrugPredictions,
  requestDiseaseOptions,
  requestEmbedding,
  requestAttentionPair,
};
