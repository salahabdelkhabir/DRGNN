import {
  DrugPrediction,
  IDispatch,
  IPath,
  IMetaPath,
  IMetaPathSummary,
} from 'types';
import {
  requestAttentionPair,
  requestDrugPredictions,
} from 'stores/DataService';
import { Descriptions } from 'antd';

export const ACTION_TYPES = {
  Load_Node_Types: 'Load_Node_Types',
  Load_Edge_Types: 'Load_Edge_Types',
  Load_Meta_Paths: 'Load_Meta_Paths',
  Add_Attention_Paths: 'Add_Attention_Paths',
  Del_Attention_Paths: 'Del_Attention_Paths',
  Load_Node_Name_Dict: 'Load_Node_Name_Dict',
  Load_Drug_Options: 'Load_Drug_Options',
  Load_Disease_Options: 'Load_Disease_Options',

  Set_Loading_Status: 'Set_Loading_Status',

  Change_Edge_THR: 'Change_Edge_THR',
  Change_Drug: 'Change_Drug',
  Change_Disease: 'Change_Disease',
  Select_Path_Noes: 'Select_Path_Nodes',

  Toggle_Meta_Path_Hide: 'Toggle_Meta_Path_Hide',
  Toggle_Meta_Path_Expand: 'Toggle_Meta_Path_Expand',
  Update_Case_Description: 'Update_Case_Description',
};

export const selectDrug = (
  selectedDrug: string,
  selectedDisease: string | undefined,
  isAdd: boolean,
  dispatch: IDispatch
) => {
  if (selectedDisease) {
    changeDrug(selectedDrug, dispatch);
    return modifyAttentionPaths(selectedDrug, selectedDisease, isAdd, dispatch);
  }
};

export const updateCaseDescription = (
  caseDescription: JSX.Element | undefined,
  dispatch: IDispatch
) => {
  dispatch({
    type: ACTION_TYPES.Update_Case_Description,
    payload: { caseDescription },
  });
};

export const selectDisease = (selectedDisease: string, dispatch: IDispatch) => {
  console.log(`Frontend - selectDisease - Selecting disease: ${selectedDisease}`);
  
  if (!selectedDisease) {
    console.warn('Frontend - selectDisease - Empty disease ID, not proceeding with selection');
    return Promise.resolve();
  }
  
  dispatch({
    type: ACTION_TYPES.Set_Loading_Status,
    payload: { isDrugLoading: true, isAttentionLoading: true },
  });

  dispatch({
    type: ACTION_TYPES.Change_Disease,
    payload: { selectedDisease },
  });

  dispatch({
    type: ACTION_TYPES.Change_Drug,
    payload: { selectedDrug: undefined },
  });

  return requestDrugPredictions(selectedDisease)
    .then((res) => {
      console.log(`Frontend - selectDisease - Received predictions, count: ${res.length}`);
      
      if (Array.isArray(res) && res.length > 0) {
        const drugPredictions = res.map((d: DrugPrediction) => {
          return { ...d, selected: false };
        });
        
        dispatch({
          type: ACTION_TYPES.Load_Drug_Options,
          payload: { drugPredictions },
        });
      } else {
        console.warn('Frontend - selectDisease - No predictions received, setting empty array');
        dispatch({
          type: ACTION_TYPES.Load_Drug_Options,
          payload: { drugPredictions: [] },
        });
      }
    })
    .catch((error) => {
      console.error('Frontend - selectDisease - Error processing predictions:', error);
      dispatch({
        type: ACTION_TYPES.Load_Drug_Options,
        payload: { drugPredictions: [] },
      });
    })
    .finally(() => {
      dispatch({
        type: ACTION_TYPES.Set_Loading_Status,
        payload: { isDrugLoading: false, isAttentionLoading: false },
      });
    });
};
const modifyAttentionPaths = (
  selectedDrug: string | undefined,
  selectedDisease: string | undefined,
  isAdd: boolean,
  dispatch: IDispatch
) => {
  if (selectedDrug !== undefined && selectedDisease !== undefined) {
    if (isAdd) {
      dispatch({
        type: ACTION_TYPES.Set_Loading_Status,
        payload: { isAttentionLoading: true },
      });

      return requestAttentionPair(selectedDisease, selectedDrug)
        .then((res) => {
          // Add validation for the response
          if (!res || !res.attention) {
            console.error('Invalid response from requestAttentionPair:', res);
            // Return empty attention data
            dispatch({
              type: ACTION_TYPES.Add_Attention_Paths,
              payload: {
                attention: {},
                selectedDrug,
                metaPathGroups: { [selectedDrug]: [] },
              },
            });
            return;
          }
          
          dispatch({
            type: ACTION_TYPES.Add_Attention_Paths,
            payload: {
              attention: res.attention,
              selectedDrug,
              metaPathGroups: { [selectedDrug]: groupMetaPaths(res.paths || []) },
            },
          });
        })
        .catch((error) => {
          console.error('Error in requestAttentionPair:', error);
          dispatch({
            type: ACTION_TYPES.Add_Attention_Paths,
            payload: {
              attention: {},
              selectedDrug,
              metaPathGroups: { [selectedDrug]: [] },
            },
          });
        })
        .finally(() => {
          dispatch({
            type: ACTION_TYPES.Set_Loading_Status,
            payload: { isAttentionLoading: false },
          });
        });
    } else {
      dispatch({
        type: ACTION_TYPES.Del_Attention_Paths,
        payload: {
          selectedDrug,
        },
      });

      return Promise.resolve();
    }
  }
  return Promise.resolve();
};

const changeDrug = (selectedDrug: string, dispatch: IDispatch) => {
  dispatch({
    type: ACTION_TYPES.Change_Drug,
    payload: { selectedDrug },
  });
};

const groupMetaPaths = (paths: IPath[]): IMetaPath[] => {
  let groups: IMetaPath[] = [];
  let groupDict: string[] = [];
  paths.forEach((path) => {
    path.hide = false; // initi, show all metapaths
    const nodeTypeString = path.nodes.map((d) => d.nodeType).join('_');
    const groupIdx = groupDict.indexOf(nodeTypeString);
    if (groupIdx > -1) {
      groups[groupIdx].paths.push(path);
    } else {
      groupDict.push(nodeTypeString);
      groups.push({
        nodeTypes: path.nodes.map((d) => d.nodeType),
        paths: [path],
      });
    }
  });
  return groups;
};

export const toggleMetaPathHide = (
  metaPathSummary: IMetaPathSummary[],
  idx: number,
  dispatch: IDispatch
) => {
  metaPathSummary[idx]['hide'] = !metaPathSummary[idx]['hide'];
  if (metaPathSummary[idx]['hide']) {
    metaPathSummary[idx]['expand'] = false; // if hide, then collapse
  }
  dispatch({
    type: ACTION_TYPES.Toggle_Meta_Path_Hide,
    payload: { metaPathSummary },
  });
};

export const toggleMetaPathExpand = (
  metaPathSummary: IMetaPathSummary[],
  idx: number,
  dispatch: IDispatch
) => {
  metaPathSummary[idx]['expand'] = !metaPathSummary[idx]['expand'];
  dispatch({
    type: ACTION_TYPES.Toggle_Meta_Path_Hide,
    payload: { metaPathSummary },
  });
};
