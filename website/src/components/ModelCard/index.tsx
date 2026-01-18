import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

interface Model {
  id: string;
  name: string;
  description: string;
  version: string;
  parameters: string | null;
  platforms: string[];
  isExperimental: boolean;
  tags: string[];
  isVision: boolean;
  huggingfaceUrl: string | null;
}

interface ModelCardProps {
  model: Model;
}

export default function ModelCard({ model }: ModelCardProps): JSX.Element {
  const [copied, setCopied] = useState(false);

  const cliCommand = `pie model add ${model.name}`;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(cliCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const getPlatformLabel = (platform: string): string => {
    switch (platform) {
      case 'cuda':
        return 'CUDA';
      case 'metal':
        return 'Metal';
      case 'desktop-macos':
        return 'Desktop (Mac)';
      default:
        return platform;
    }
  };

  return (
    <article className={styles.modelCard}>
      {/* Model Name */}
      <h3 className={styles.modelName}>{model.name}</h3>

      {/* Model Description */}
      <p className={styles.modelDescription}>{model.description}</p>

      {/* Model Info: Parameters + Platforms/Experimental */}
      <div className={styles.modelInfo}>
        {model.parameters && (
          <span className={styles.parameterBadge}>{model.parameters}</span>
        )}

        {/* Platform badges or Experimental */}
        {model.isExperimental ? (
          <span className={clsx(styles.badge, styles.experimentalBadge)}>
            Experimental
          </span>
        ) : (
          model.platforms.map(platform => (
            <span key={platform} className={clsx(styles.badge, styles.platformBadge)}>
              {getPlatformLabel(platform)}
            </span>
          ))
        )}

        {/* Vision badge */}
        {model.isVision && (
          <span className={clsx(styles.badge, styles.visionBadge)}>
            Vision
          </span>
        )}
      </div>

      {/* CLI Command */}
      <div className={styles.cliSection}>
        <code className={styles.cliCommand}>$ {cliCommand}</code>
        <button
          className={styles.copyButton}
          onClick={handleCopy}
          aria-label="Copy command to clipboard"
          title="Copy to clipboard"
        >
          {copied ? '✓' : '📋'}
        </button>
      </div>

      {/* Hugging Face Link */}
      {model.huggingfaceUrl && (
        <a
          href={model.huggingfaceUrl}
          className={styles.hfLink}
          target="_blank"
          rel="noopener noreferrer"
        >
          → View on Hugging Face
        </a>
      )}
    </article>
  );
}
