import path from 'node:path';
import { finalizePresentation } from '/Users/ricardo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1110.11613/skills/presentations/container_tools/artifact_tool_utils.mjs';
const workspaceDir='/Users/ricardo/Desktop/CHATBOT(E)';
const SKILL_DIR='/Users/ricardo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1110.11613/skills/presentations';
const result=await finalizePresentation({
 workspaceDir,
 candidatePath:path.join(workspaceDir,'.ppt-academica-v2/candidate.pptx'),
 finalPath:path.join(workspaceDir,'.ppt-output/academica-v2-validada.pptx'),
 pythonExecutable:'/Users/ricardo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','5','--require-native-table-slide','10','--require-native-table-slide','14','--require-native-table-slide','15'],
 explicitTotalSlideCount:16,
 requiredNativeTableOwnerSlides:[5,10,14,15],
 requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Helvetica Neue','Menlo']},
 verifyArtifactToolImport:true,
 receiptPath:path.join(workspaceDir,'.ppt-academica-v2/validation.json')
});
console.log(JSON.stringify(result));
