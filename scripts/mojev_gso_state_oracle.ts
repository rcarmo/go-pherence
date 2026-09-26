/** Generate model-free state-render observations using pinned GSO code via a Go overlay.
 * No model, GPU or service runs. Arguments: GSO_CHECKOUT OUTPUT_DIR
 */
import {resolve} from 'node:path';
import {realpathSync} from 'node:fs';
const [repoArg,outArg]=process.argv.slice(2);if(!repoArg||!outArg)throw Error('usage: bun scripts/mojev_gso_state_oracle.ts GSO_CHECKOUT OUTPUT_DIR');
const repo=realpathSync(repoArg),out=resolve(outArg);
const sha='1cd60986f1a3e37d07f7743e9e126d26a96f54047527836062b453e410b7fde8';
const source=await Bun.file(repo+'/model/gosystemone/systemone.go').arrayBuffer();
if(new Bun.CryptoHasher('sha256').update(source).digest('hex')!==sha)throw Error('GSO renderer source mismatch');
const cases=[
 ['text','"  café <literal>  "'],['object','{ "z": 1.2300e+02, "a": [true, null, {"x":"café"}] }'],
 ['array','[ "x", 9007199254740993, -0.0, {} ]'],['null','null'],['escapes','{"\\u0062":"\\u003cmarker\\u003e","line":"a\\nb"}'],
 ['empty-object','{}'],['empty-array','[]'],['bool','true'],['number','42'],['two-values','{} []']
].map(([name,raw])=>({name,raw}));
const test=`package gosystemone
import("encoding/json";"os";"testing")
func TestMoJevStateOracle(t *testing.T){
 var cases []struct{Name string \`json:"name"\`;Raw string \`json:"raw"\`}
 if err:=json.Unmarshal([]byte(${JSON.stringify(JSON.stringify(cases))}),&cases);err!=nil{t.Fatal(err)}
 type observation struct{Name string \`json:"name"\`;Raw string \`json:"raw"\`;Text string \`json:"text"\`;Valid bool \`json:"valid"\`}
 observations:=make([]observation,0,len(cases))
 for _,c:=range cases{text,err:=entryText(json.RawMessage(c.Raw),false);observations=append(observations,observation{c.Name,c.Raw,text,err==nil})}
 result:=struct{Schema int \`json:"schema"\`;SourceRevision string \`json:"source_revision"\`;SourceSHA string \`json:"source_sha256"\`;Cases []observation \`json:"cases"\`}{1,"b18ee0d4748bac436999aa72c000e06406c3cce6","${sha}",observations}
 data,err:=json.MarshalIndent(result,"","  ");if err!=nil{t.Fatal(err)};if err=os.WriteFile(os.Getenv("MOJEV_GSO_STATE_ORACLE"),append(data,'\\n'),0600);err!=nil{t.Fatal(err)}
}
`;
await Bun.write(out+'/oracle_test.go',test);
await Bun.write(out+'/overlay.json',JSON.stringify({Replace:{[repo+'/model/gosystemone/systemone_test.go']:out+'/oracle_test.go'}},null,2));
const proc=Bun.spawn(['go','test','-overlay',out+'/overlay.json','./model/gosystemone','-run','^TestMoJevStateOracle$','-count=1'],{cwd:repo,env:{...process.env,GOMAXPROCS:'6',GO_PHERENCE_DISABLE_NVIDIA:'1',MOJEV_GSO_STATE_ORACLE:out+'/fixture.json'},stdout:'inherit',stderr:'inherit'});
if(await proc.exited!==0 || !await Bun.file(out+'/fixture.json').exists())throw Error('GSO oracle failed or did not execute');
