import { ipcRenderer } from 'electron';
import {
  DEEPDANBOORU_PREDICT,
  DEEPDANBOORU_PREDICT_REPLY,
  DeepDanbooruPredictMessage,
  DeepDanbooruPredictReply,
} from '../ipc/messages';

export const DEFAULT_TAG_THRESHOLD = 0.5;

export async function predictDeepDanbooru(
  imagePath: string,
  projectPath: string,
  threshold = 0.5,
): Promise<DeepDanbooruPredictReply> {
  let useProject = projectPath;
  if (!useProject) {
    // ask main process for configured DeepDanbooru project location
    try {
      // deepdanbooru:get-config returns an object with optional projectPath/modelPath/tagListPath
      // prefer a projectPath if present
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      // use ipcRenderer.invoke to fetch config
      // NOTE: this may return {} if not configured
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const cfg: any = await (ipcRenderer as any).invoke('deepdanbooru:get-config');
      if (cfg && cfg.projectPath) useProject = cfg.projectPath;
    } catch (e) {
      console.warn('Could not read DeepDanbooru config from main process', e);
    }
  }
  const msg: DeepDanbooruPredictMessage = { imagePath, projectPath: useProject, threshold };
  return new Promise((resolve) => {
    ipcRenderer.once(DEEPDANBOORU_PREDICT_REPLY, (_, reply: DeepDanbooruPredictReply) =>
      resolve(reply),
    );
    ipcRenderer.send(DEEPDANBOORU_PREDICT, msg);
  });
}

export type { DeepDanbooruPredictMessage, DeepDanbooruPredictReply };
