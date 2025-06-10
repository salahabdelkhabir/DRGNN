import React from 'react';
import { Table, Tag, Collapse } from 'antd';
import { IState, IDispatch, IMetaPathResult } from 'types';
import { StateConsumer } from 'stores';
import { getNodeColor } from 'helpers';

const { Panel } = Collapse;

interface Props {
  globalState: IState;
  dispatch: IDispatch;
  diseaseId: string;
  drugId: string;
}

class DetailedMetaPaths extends React.Component<Props> {
  componentDidMount() {
    // You could implement loading logic here if needed
  }

  render() {
    const { globalState, diseaseId, drugId } = this.props;
    const { metaPathsCache, nodeNameDict } = globalState;
    
    // Create the key for the meta-paths cache
    const cacheKey = `${diseaseId}-${drugId}`;
    
    // Get the meta-paths from the cache
    const metaPaths = metaPathsCache?.[cacheKey] || [];
    
    if (metaPaths.length === 0) {
      return (
        <div style={{ padding: '20px', textAlign: 'center' }}>
          No meta paths found for this disease-drug pair.
        </div>
      );
    }
    
    return (
      <div className="detailed-meta-paths">
        <h3>Meta Paths Between {nodeNameDict?.disease?.[diseaseId] || diseaseId} and {nodeNameDict?.drug?.[drugId] || drugId}</h3>
        
        <Collapse defaultActiveKey={['0']}>
          {metaPaths.map((metaPath, index) => (
            <Panel 
              header={
                <div>
                  <span style={{ fontWeight: 'bold' }}>{metaPath.meta_path}</span>
                  <span style={{ marginLeft: '15px' }}>
                    Count: <Tag color="blue">{metaPath.count}</Tag>
                    Avg Score: <Tag color="green">{metaPath.avg_score.toFixed(4)}</Tag>
                  </span>
                </div>
              } 
              key={index.toString()}
            >
              <Table 
                dataSource={metaPath.paths.map((path, i) => ({ 
                  key: i, 
                  path: path.path,
                  score: path.score
                }))}
                columns={[
                  {
                    title: 'Path',
                    dataIndex: 'path',
                    key: 'path',
                    render: (text) => (
                      <div style={{ 
                        maxWidth: '800px', 
                        wordWrap: 'break-word',
                        lineHeight: '1.5'
                      }}>
                        {text.split(' -> ').map((segment: string, i: number) => {
                          if (segment.includes(':')) {
                            // This is a node
                            const [nodeType, nodeId] = segment.split(':');
                            return (
                              <Tag 
                                color={getNodeColor(nodeType)} 
                                key={i}
                                style={{ margin: '2px' }}
                              >
                                {segment}
                              </Tag>
                            );
                          } else {
                            // This is an edge
                            return (
                              <span key={i} style={{ margin: '0 5px' }}>
                                → {segment} →
                              </span>
                            );
                          }
                        })}
                      </div>
                    )
                  },
                  {
                    title: 'Score',
                    dataIndex: 'score',
                    key: 'score',
                    render: (score) => score.toFixed(4)
                  }
                ]}
                pagination={{ pageSize: 5 }}
              />
            </Panel>
          ))}
        </Collapse>
      </div>
    );
  }
}

export default StateConsumer(DetailedMetaPaths);