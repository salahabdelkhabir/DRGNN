import { IAttentionTree } from 'types';
export { cropText, getTextWidth } from './text';
export * from './icon';
export { CASES } from './cases';

export {
  setNodeColor,
  getNodeColor,
  HIGHLIGHT_COLOR,
  SELECTED_COLOR,
} from './color';

export const pruneEdge = (
  node: IAttentionTree,
  threshold: number,
  childeNum?: number
): IAttentionTree => {
  // Add a null check to handle undefined children
  if (node && node.children && node.children.length > 0) {
    node = {
      ...node,
      children: node.children
        .filter((d) => d.score >= threshold)
        .map((node) => pruneEdge(node, threshold, childeNum))
        .slice(0, childeNum), // only keep top 7 children
    };
  } else {
    // Ensure node always has a children array
    node = {
      ...node,
      children: []
    };
  }
  return node;
};

export const flatTree = (node: IAttentionTree): string[] => {
  let res = [node.nodeId];
  if (node.children.length > 0) {
    res = res.concat(...node.children.map((d) => flatTree(d)));
  }
  return res;
};

export const sigmoid = (t: number) => {
  return 1 / (1 + Math.pow(Math.E, -t));
};

export const removeDiseaseList = [
  'mendelian disease',
  'disease of cell nucleous',
  'hip region disease',
  'acute disease',
  'vector borne disease',
  'cancer',
  'sex-linked disease',
  'movement disorder',
];

// Modified wheterRemoveDisease function to handle undefined values
export const wheterRemoveDisease = (disease: string | undefined): boolean => {
  if (!disease) return false;
  return removeDiseaseList.includes(disease.toLowerCase());
};

export const sentenceCapitalizer = (str: string) => {
  return str.charAt(0).toUpperCase() + str.slice(1);
};
export const INIT_DISEASE = '';
export const INIT_DRUGS: string[] = [];