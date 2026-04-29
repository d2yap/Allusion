import { getThumbnailPath, isDirEmpty } from 'common/fs';
import { runInAction } from 'mobx';
import { observer } from 'mobx-react-lite';
import React, { useEffect, useState } from 'react';
import FileInput from 'src/frontend/components/FileInput';
import { RendererMessenger } from 'src/ipc/renderer';
import { Button, ButtonGroup, IconSet } from 'widgets';
import { useStore } from '../../contexts/StoreContext';
import { moveThumbnailDir } from '../../image/ThumbnailGeneration';
import { ClearDbButton } from '../ErrorBoundary';
import { Callout } from 'widgets/notifications';
import ExternalLink from 'src/frontend/components/ExternalLink';
import { ipcRenderer } from 'electron';

export const Advanced = observer(() => {
  const { uiStore, fileStore } = useStore();
  const thumbnailDirectory = uiStore.thumbnailDirectory;

  const [defaultThumbnailDir, setDefaultThumbnailDir] = useState('');
  useEffect(() => {
    RendererMessenger.getDefaultThumbnailDirectory().then(setDefaultThumbnailDir);
  }, []);

  const changeThumbnailDirectory = async (newDir: string) => {
    const oldDir = thumbnailDirectory;

    // Move thumbnail files
    await moveThumbnailDir(oldDir, newDir);
    uiStore.setThumbnailDirectory(newDir);

    // Reset thumbnail paths for those that already have one
    runInAction(() => {
      for (const f of fileStore.fileList) {
        if (f.thumbnailPath && f.thumbnailPath !== f.absolutePath) {
          f.setThumbnailPath(getThumbnailPath(f.absolutePath, newDir));
        }
      }
    });
  };

  const browseThumbnailDirectory = async ([newDir]: [string, ...string[]]) => {
    if (!(await isDirEmpty(newDir))) {
      if (
        window.confirm(
          `The directory you picked is not empty. Allusion might delete any files inside of it. Do you still wish to pick this directory?\n\nYou picked: ${newDir}`,
        )
      ) {
        changeThumbnailDirectory(newDir);
      }
    } else {
      changeThumbnailDirectory(newDir);
    }
  };

  return (
    <>
      {/* Todo: Add support to toggle this */}
      {/* <Switch checked={true} onChange={() => alert('Not supported yet')} label="Generate thumbnails" /> */}
      <Callout icon={IconSet.INFO}>
        By default thumbnails are stored in the{' '}
        <ExternalLink url={defaultThumbnailDir}>temporary files directory</ExternalLink>.
      </Callout>
      <div className="filepicker">
        <FileInput
          className="btn-minimal filepicker-input"
          options={{
            properties: ['openDirectory'],
            defaultPath: thumbnailDirectory,
          }}
          onChange={browseThumbnailDirectory}
        >
          Change...
        </FileInput>
        <h3 className="filepicker-label">Thumbnail Directory</h3>
        <div className="filepicker-path">{thumbnailDirectory}</div>
      </div>

      <h3>Auto-tagging</h3>
      <div style={{ marginBottom: 8 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>Tagger model</label>
        <div className="filepicker">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input readOnly value={uiStore.deepDanbooruModelPath} className="filepicker-path" />
            <FileInput
              className="btn-minimal filepicker-input"
              options={{
                properties: ['openFile'],
                filters: [{ name: 'ONNX / Keras model', extensions: ['onnx', 'h5'] }],
              }}
              onChange={async ([p]) => {
                if (!p) {
                  return;
                }
                // persist config via main
                await (ipcRenderer as any).invoke('deepdanbooru:set-config', { modelPath: p });
                uiStore.deepDanbooruModelPath = p;
              }}
            >
              Choose...
            </FileInput>
          </div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>WD14 tag list</label>
        <div className="filepicker">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input readOnly value={uiStore.deepDanbooruTagListPath} className="filepicker-path" />
            <FileInput
              className="btn-minimal filepicker-input"
              options={{
                properties: ['openFile'],
                filters: [{ name: 'Tag list', extensions: ['csv', 'txt', 'json'] }],
              }}
              onChange={async ([p]) => {
                if (!p) {
                  return;
                }
                await (ipcRenderer as any).invoke('deepdanbooru:set-config', { tagListPath: p });
                uiStore.deepDanbooruTagListPath = p;
              }}
            >
              Choose...
            </FileInput>
          </div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <input
            type="checkbox"
            checked={uiStore.isAutoTagOnImportEnabled}
            onChange={(e) => uiStore.setAutoTagOnImport(e.target.checked)}
          />
          Automatically tag untagged images on import/startup
        </label>
      </div>
      <div style={{ marginTop: 8 }}>
        <button className="btn" type="button" onClick={() => fileStore.autoTagUntagged()}>
          Run auto-tag scan now
        </button>
      </div>

      <h3>Development</h3>
      <ButtonGroup>
        <ClearDbButton />
        <Button
          onClick={RendererMessenger.toggleDevTools}
          styling="outlined"
          icon={IconSet.CHROME_DEVTOOLS}
          text="Toggle DevTools"
        />
      </ButtonGroup>
    </>
  );
});
