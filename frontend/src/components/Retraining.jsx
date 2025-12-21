import React, { useState, useEffect, useRef } from 'react';
import {
    Box, Typography, Button, Paper, Grid, Alert, LinearProgress, Stack,
    FormControl, InputLabel, Select, MenuItem, Slider, Tooltip, IconButton,
    Card, CardContent, TextField, Collapse, Radio, RadioGroup, FormControlLabel, FormLabel, Snackbar, AlertTitle,
    Tabs, Tab, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AutoModeIcon from '@mui/icons-material/AutoMode';
import TuneIcon from '@mui/icons-material/Tune';
import InfoIcon from '@mui/icons-material/Info';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TerminalIcon from '@mui/icons-material/Terminal';
import axios from 'axios';

const MODEL_INFO = {
    "Random Forest": "An ensemble learning method that constructs multiple decision trees. Good for handling non-linear data and reducing overfitting.",
    "XGBoost": "Extreme Gradient Boosting. Highly efficient and flexible. Great for structured data and often achieves state-of-the-art results.",
    "LightGBM": "Light Gradient Boosting Machine. Faster training speed and lower memory usage. efficient for large datasets.",
    "Gradient Boosting": "Builds models sequentially, with each new model correcting errors of the previous ones.",
    "SVM (SVR)": "Support Vector Regression. Effective in high-dimensional spaces. Uses kernels to handle non-linear relationships.",
    "Gaussian Process": "A probabilistic model that provides uncertainty estimates. Excellent for small datasets but computationally expensive.",
    "Kernel Ridge": "Combines Ridge Regression with the kernel trick. Similar to SVR but uses mean squared error loss. Good for non-linear regression."
};

const PARAM_GUIDE = {
    "n_estimators": {
        desc: "Number of trees in the forest/ensemble.",
        effect: "Increasing this generally improves performance and robustness but linearly increases training time. \n\nLogic: More trees = more 'votes' to average out errors. Too few may underfit; too many yields diminishing returns."
    },
    "learning_rate": {
        desc: "Step size shrinkage used in update to prevent overfitting.",
        effect: "Lower values make the model more robust but slower to train (requires more trees). \n\nLogic: Controls how much the model learns from the errors of the previous tree. Small steps = precise convergence."
    },
    "max_depth": {
        desc: "Maximum depth of a tree.",
        effect: "Higher depth captures more complex patterns but increases risk of overfitting (high variance). \n\nLogic: Deep trees memorize data; shallow trees generalize better but might miss fine details."
    },
    "C": {
        desc: "Regularization parameter for SVM.",
        effect: "High C = stricter margin (tries to classify all training points correctly, risk of overfitting). \nLow C = softer margin (allows more errors to get a smoother boundary)."
    },
    "epsilon": {
        desc: "Epsilon tube width for SVR.",
        effect: "Defines a margin of tolerance where no penalty is given to errors. \n\nLogic: Larger epsilon = more tolerance (sparser model). Smaller epsilon = stricter fit."
    },
    "alpha": {
        desc: "Regularization strength (Noise level for GP, Penalty for KRR).",
        effect: "Controls the complexity (smoothness) of the model. Higher values = smoother function (less overfitting)."
    },
    "gamma": {
        desc: "Kernel coefficient.",
        effect: "Defines how far the influence of a single training example reaches. \nHigh Gamma = Close reach (complex, wiggly boundary). \nLow Gamma = Far reach (smooth boundary)."
    },
    "n_restarts_optimizer": {
        desc: "Optimizer restarts for GP.",
        effect: "Number of times to restart the internal optimizer to find the best kernel parameters. \n\nLogic: Helps avoid local minima but increases training time."
    }
};

const TrainingConfigPanel = ({ config, setConfig, label }) => {

    const updateConfig = (key, value) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    };

    const renderParamSlider = (name, value, key, min, max, step) => (
        <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography id={`slider-${label}-${key}`} gutterBottom variant="body2" sx={{ fontWeight: 500 }}>
                    {name}: {value}
                </Typography>
                <Tooltip
                    title={<div style={{ whiteSpace: 'pre-line' }}>{PARAM_GUIDE[key]?.desc} <br /><br /> <strong>Effect:</strong> {PARAM_GUIDE[key]?.effect}</div>}
                    arrow
                    placement="right"
                >
                    <IconButton size="small" sx={{ ml: 1, color: 'primary.main' }}>
                        <InfoIcon fontSize="small" />
                    </IconButton>
                </Tooltip>
            </Box>
            <Slider
                value={value}
                min={min}
                max={max}
                step={step}
                onChange={(e, val) => updateConfig(key, val)}
                valueLabelDisplay="auto"
                aria-labelledby={`slider-${label}-${key}`}
                size="small"
                sx={{ width: '95%' }}
            />
        </Box>
    );

    return (
        <Box sx={{ p: 2, border: '1px solid #eee', borderRadius: 2 }}>
            <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel size="small">Model Architecture</InputLabel>
                <Select
                    value={config.model}
                    label="Model Architecture"
                    size="small"
                    onChange={(e) => updateConfig('model', e.target.value)}
                >
                    {Object.keys(MODEL_INFO).map(model => (
                        <MenuItem key={model} value={model}>{model}</MenuItem>
                    ))}
                </Select>
                <Typography variant="caption" sx={{ mt: 1, color: 'text.secondary', display: 'block', minHeight: '3em' }}>
                    {MODEL_INFO[config.model]}
                </Typography>
            </FormControl>

            {['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting'].includes(config.model) && (
                <>
                    {renderParamSlider("Number of Estimators", config.n_estimators, "n_estimators", 10, 1000, 10)}
                    {renderParamSlider("Max Tree Depth", config.max_depth, "max_depth", 1, 50, 1)}
                </>
            )}

            {['XGBoost', 'LightGBM', 'Gradient Boosting'].includes(config.model) && (
                <>
                    {renderParamSlider("Learning Rate", config.learning_rate, "learning_rate", 0.001, 0.5, 0.001)}
                </>
            )}

            {config.model === 'SVM (SVR)' && (
                <>
                    {renderParamSlider("C (Regularization)", config.C, "C", 0.1, 1000, 0.1)}
                    {renderParamSlider("Epsilon", config.epsilon, "epsilon", 0.001, 1, 0.001)}
                </>
            )}

            {config.model === 'Gaussian Process' && (
                <>
                    {renderParamSlider("Optimizer Restarts", config.n_restarts_optimizer || 0, "n_restarts_optimizer", 0, 10, 1)}
                    {renderParamSlider("Alpha (Noise)", config.alpha || 1e-10, "alpha", 1e-10, 1.0, 0.0001)}
                </>
            )}

            {config.model === 'Kernel Ridge' && (
                <>
                    {renderParamSlider("Alpha (Regularization)", config.alpha || 1.0, "alpha", 0.0001, 5.0, 0.01)}
                    {renderParamSlider("Gamma (Kernel)", config.gamma || 0.1, "gamma", 0.001, 5.0, 0.001)}
                </>
            )}
        </Box>
    );
};

const Retraining = ({ isTraining, progress, statusMessage, trainingLogs = [], errorState }) => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState(null);
    const [uploadError, setUploadError] = useState(null);
    const [stopping, setStopping] = useState(false);

    // Error Dialog State
    const [errorDialogOpen, setErrorDialogOpen] = useState(false);

    const logsContainerRef = useRef(null);
    const lastErrorRef = useRef(null); // Track the last error to prevent re-opening

    // Watch for backend errors
    useEffect(() => {
        if (errorState && JSON.stringify(errorState) !== lastErrorRef.current) {
            setErrorDialogOpen(true);
            setStopping(false); // If error occurred, we are no longer "stopping"
            lastErrorRef.current = JSON.stringify(errorState);
        }
    }, [errorState]);

    // Auto-scroll logs (only scroll the container, not the window)
    // Auto-scroll logs (Smart Scroll)
    useEffect(() => {
        const container = logsContainerRef.current;
        if (container) {
            // Check if user is scrolled to the bottom (with a small buffer of 50px)
            const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50;

            // Only auto-scroll if they were ALREADY at the bottom or if it's the very first log
            if (isNearBottom || trainingLogs.length < 5) {
                container.scrollTop = container.scrollHeight;
            }
        }
    }, [trainingLogs]);

    const handleCloseErrorDialog = () => setErrorDialogOpen(false);

    // ... existing state ...



    const [mode, setMode] = useState('auto'); // 'auto' or 'manual'
    const [trainingMode, setTrainingMode] = useState('standard');
    const [isConsoleOpen, setIsConsoleOpen] = useState(true); // Default open when training

    // Auto Config Granular State
    const [selectionStrategy, setSelectionStrategy] = useState('global'); // 'global', 'independent'
    const [autoGlobalModel, setAutoGlobalModel] = useState('Auto Select');
    const [d33Model, setD33Model] = useState('Auto Select');
    const [tcModel, setTcModel] = useState('Auto Select');

    // Manual Config State - GRANULAR
    const [manualStrategy, setManualStrategy] = useState('independent'); // 'global' (legacy) or 'independent'
    const [manualTab, setManualTab] = useState(0);

    const defaultConfig = {
        model: 'XGBoost',
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 5,
        C: 100,
        epsilon: 0.1
    };

    const [d33ManualConfig, setD33ManualConfig] = useState({ ...defaultConfig });
    const [tcManualConfig, setTcManualConfig] = useState({ ...defaultConfig });
    const [globalManualConfig, setGlobalManualConfig] = useState({ ...defaultConfig });

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
        setMessage(null);
        setUploadError(null);
    };

    const handleTrain = async () => {
        if (!file) {
            setUploadError("Please upload a dataset CSV file.");
            return;
        }

        setMessage(null);
        setUploadError(null);
        setStopping(false);
        setIsConsoleOpen(true); // Auto open console

        const formData = new FormData();
        formData.append('file', file);
        formData.append('mode', mode === 'auto' ? 'Auto' : 'Manual_Granular');
        formData.append('training_mode', trainingMode);

        if (mode === 'auto') {
            if (selectionStrategy === 'global') {
                const type = autoGlobalModel === 'Auto Select' ? 'Auto' : autoGlobalModel;
                formData.append('model_type', type);
                formData.append('d33_model_type', type);
                formData.append('tc_model_type', type);
            } else {
                formData.append('d33_model_type', d33Model === 'Auto Select' ? 'Auto' : d33Model);
                formData.append('tc_model_type', tcModel === 'Auto Select' ? 'Auto' : tcModel);
            }
            formData.append('auto_tune', 'true');
        } else {
            // Manual Mode
            formData.append('model_type', 'Manual_Granular'); // Signal backend to look at params

            if (manualStrategy === 'global') {
                // Apply global config to both
                formData.append('d33_model_type', globalManualConfig.model);
                formData.append('tc_model_type', globalManualConfig.model);
                formData.append('d33_params', JSON.stringify(globalManualConfig));
                formData.append('tc_params', JSON.stringify(globalManualConfig));
            } else {
                // Independent
                formData.append('d33_model_type', d33ManualConfig.model);
                formData.append('tc_model_type', tcManualConfig.model);
                formData.append('d33_params', JSON.stringify(d33ManualConfig));
                formData.append('tc_params', JSON.stringify(tcManualConfig));
            }
        }

        try {
            await axios.post('http://localhost:8000/train', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setMessage("Training started successfully! You can monitor progress below.");
        } catch (err) {
            console.error(err);
            setUploadError(err.response?.data?.detail || "Failed to initiate training. Ensure server is running.");
        }
    };

    const handleStop = async () => {
        setStopping(true);
        try {
            await axios.post('http://localhost:8000/stop-training');
            setMessage("Stop signal sent. Reverting changes...");
        } catch (err) {
            console.error(err);
            setUploadError("Failed to stop training.");
            setStopping(false);
        }
    };



    const renderManualSection = () => (
        <Box sx={{ mb: 4, textAlign: 'left', maxWidth: 800, mx: 'auto', border: '1px solid #e0e0e0', p: 3, borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom color="primary">
                Expert Manual Configuration
            </Typography>

            <FormControl component="fieldset" sx={{ mb: 2, width: '100%' }}>
                <FormLabel component="legend" sx={{ fontWeight: 'bold', mb: 1 }}>Configuration Strategy</FormLabel>
                <RadioGroup
                    row
                    value={manualStrategy}
                    onChange={(e) => setManualStrategy(e.target.value)}
                    sx={{ bgcolor: '#f5f5f5', p: 2, borderRadius: 2, mb: 2 }}
                >
                    <FormControlLabel value="global" control={<Radio />} label="Unified (Apply to All)" />
                    <FormControlLabel value="independent" control={<Radio />} label="Independent (Fine-Tune Separately)" />
                </RadioGroup>
            </FormControl>

            {manualStrategy === 'global' ? (
                <TrainingConfigPanel
                    config={globalManualConfig}
                    setConfig={setGlobalManualConfig}
                    label="Global"
                />
            ) : (
                <Box>
                    <Tabs value={manualTab} onChange={(e, v) => setManualTab(v)} sx={{ mb: 3, borderBottom: 1, borderColor: 'divider' }}>
                        <Tab label="d33 Configuration" />
                        <Tab label="Tc Configuration" />
                    </Tabs>

                    {manualTab === 0 && (
                        <TrainingConfigPanel
                            config={d33ManualConfig}
                            setConfig={setD33ManualConfig}
                            label="d33"
                        />
                    )}
                    {manualTab === 1 && (
                        <TrainingConfigPanel
                            config={tcManualConfig}
                            setConfig={setTcManualConfig}
                            label="Tc"
                        />
                    )}
                </Box>
            )}
        </Box>
    );

    return (
        <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" component="h2" gutterBottom color="primary" sx={{ fontWeight: 'bold' }}>
                Model Retraining Laboratory
            </Typography>
            <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 4, maxWidth: 800, mx: 'auto' }}>
                Update the AI models with new experimental data. Upload a CSV file containing 'Component', 'd33 (pC/N)', and 'Tc (°C)' columns.
            </Typography>

            <Grid container spacing={4} justifyContent="center">
                <Grid item xs={12} md={10}>
                    <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>

                        <Stack direction="row" spacing={2} justifyContent="center" sx={{ mb: 4 }}>
                            <Button
                                variant={mode === 'auto' ? "contained" : "outlined"}
                                onClick={() => setMode('auto')}
                                startIcon={<AutoModeIcon />}
                                sx={{ borderRadius: 20, px: 4 }}
                                disabled={isTraining}
                            >
                                Intelligent Auto-Tune
                            </Button>
                            <Button
                                variant={mode === 'manual' ? "contained" : "outlined"}
                                onClick={() => setMode('manual')}
                                startIcon={<TuneIcon />}
                                sx={{ borderRadius: 20, px: 4 }}
                                disabled={isTraining}
                            >
                                Manual Configuration
                            </Button>
                        </Stack>

                        {mode === 'auto' && (
                            <Box sx={{ mb: 4, textAlign: 'left', maxWidth: 800, mx: 'auto', border: '1px solid #e0e0e0', p: 3, borderRadius: 2 }}>
                                <Typography variant="h6" gutterBottom color="primary">
                                    Intelligent Auto-Tune Configuration
                                </Typography>

                                <FormControl component="fieldset" sx={{ mb: 4, width: '100%' }}>
                                    <FormLabel component="legend" sx={{ fontWeight: 'bold', mb: 1 }}>1. Optimization Intensity</FormLabel>
                                    <RadioGroup
                                        row
                                        value={trainingMode}
                                        onChange={(e) => setTrainingMode(e.target.value)}
                                        sx={{ bgcolor: '#f5f5f5', p: 2, borderRadius: 2 }}
                                    >
                                        <FormControlLabel value="standard" control={<Radio />} label="Standard (Fast)" />
                                        <FormControlLabel value="accuracy" control={<Radio />} label="Maximum Accuracy (Extensive Search)" />
                                    </RadioGroup>
                                </FormControl>

                                <FormControl component="fieldset" sx={{ mb: 2, width: '100%' }}>
                                    <FormLabel component="legend" sx={{ fontWeight: 'bold', mb: 1 }}>2. Model Selection Strategy</FormLabel>
                                    <RadioGroup
                                        row
                                        value={selectionStrategy}
                                        onChange={(e) => setSelectionStrategy(e.target.value)}
                                        sx={{ bgcolor: '#f5f5f5', p: 2, borderRadius: 2, mb: 2 }}
                                    >
                                        <FormControlLabel value="global" control={<Radio />} label="Global Configuration (Apply to All)" />
                                        <FormControlLabel value="independent" control={<Radio />} label="Independent (Separately for d33/Tc)" />
                                    </RadioGroup>
                                </FormControl>

                                {selectionStrategy === 'global' ? (
                                    <FormControl fullWidth>
                                        <InputLabel>Target Model (All Properties)</InputLabel>
                                        <Select
                                            value={autoGlobalModel}
                                            label="Target Model (All Properties)"
                                            onChange={(e) => setAutoGlobalModel(e.target.value)}
                                            disabled={isTraining}
                                        >
                                            <MenuItem value="Auto Select">Auto Select (AI Recommends Best)</MenuItem>
                                            {Object.keys(MODEL_INFO).map(model => (
                                                <MenuItem key={model} value={model}>{model}</MenuItem>
                                            ))}
                                        </Select>
                                        <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                                            The system will optimize the selected model (or find the best one) for both d33 and Tc.
                                        </Typography>
                                    </FormControl>
                                ) : (
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} md={6}>
                                            <FormControl fullWidth>
                                                <InputLabel>d33 Model</InputLabel>
                                                <Select
                                                    value={d33Model}
                                                    label="d33 Model"
                                                    onChange={(e) => setD33Model(e.target.value)}
                                                    disabled={isTraining}
                                                >
                                                    <MenuItem value="Auto Select">Auto Select</MenuItem>
                                                    {Object.keys(MODEL_INFO).map(model => (
                                                        <MenuItem key={model} value={model}>{model}</MenuItem>
                                                    ))}
                                                </Select>
                                            </FormControl>
                                        </Grid>
                                        <Grid item xs={12} md={6}>
                                            <FormControl fullWidth>
                                                <InputLabel>Tc Model</InputLabel>
                                                <Select
                                                    value={tcModel}
                                                    label="Tc Model"
                                                    onChange={(e) => setTcModel(e.target.value)}
                                                    disabled={isTraining}
                                                >
                                                    <MenuItem value="Auto Select">Auto Select</MenuItem>
                                                    {Object.keys(MODEL_INFO).map(model => (
                                                        <MenuItem key={model} value={model}>{model}</MenuItem>
                                                    ))}
                                                </Select>
                                            </FormControl>
                                        </Grid>
                                    </Grid>
                                )}
                            </Box>
                        )}

                        {mode === 'manual' && renderManualSection()}

                        <Box
                            sx={{
                                border: '2px dashed #ccc',
                                borderRadius: 4,
                                p: 4,
                                mb: 4,
                                bgcolor: 'background.default',
                                cursor: isTraining ? 'not-allowed' : 'pointer',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                    bgcolor: isTraining ? 'background.default' : 'action.hover',
                                    borderColor: isTraining ? '#ccc' : 'primary.main',
                                    transform: isTraining ? 'none' : 'scale(1.02)'
                                },
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                minHeight: 200,
                                opacity: isTraining ? 0.6 : 1
                            }}
                            component="label"
                        >
                            <input
                                type="file"
                                hidden
                                accept=".csv"
                                onChange={handleFileChange}
                                disabled={isTraining}
                            />
                            <CloudUploadIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                            <Typography variant="h6" color="text.primary" sx={{ wordBreak: 'break-word', px: 2 }}>
                                {file ? file.name : "Click to Upload Dataset (CSV)"}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Drag and drop or browse
                            </Typography>
                        </Box>

                        {/* Progress Section - Shows when training OR whenever relevant status exists */}
                        {isTraining && (
                            <Box sx={{ width: '100%', mb: 3 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                    <Typography variant="body2" color="primary" fontWeight="bold">
                                        {statusMessage || "Initializing..."}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        {progress}%
                                    </Typography>
                                </Box>
                                <LinearProgress variant="determinate" value={progress} sx={{ height: 10, borderRadius: 5 }} />
                            </Box>
                        )}

                        {!isTraining && progress === 100 && (
                            <Alert severity="success" sx={{ mb: 3 }}>Training successfully completed! You can now use the new models.</Alert>
                        )}

                        {message && !isTraining && progress !== 100 && <Alert severity="info" sx={{ mb: 3 }}>{message}</Alert>}
                        {uploadError && <Alert severity="error" sx={{ mb: 3 }}>{uploadError}</Alert>}

                        <Dialog
                            open={errorDialogOpen}
                            onClose={handleCloseErrorDialog}
                            aria-labelledby="alert-dialog-title"
                            aria-describedby="alert-dialog-description"
                        >
                            <DialogTitle id="alert-dialog-title" color="error">
                                Training Failed
                            </DialogTitle>
                            <DialogContent>
                                <Box id="alert-dialog-description">
                                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                                        What happened:
                                    </Typography>
                                    <Typography variant="body2" paragraph sx={{ bgcolor: '#ffebee', p: 1, borderRadius: 1 }}>
                                        {errorState?.message || "An unexpected error occurred."}
                                    </Typography>

                                    {errorState?.details && (
                                        <>
                                            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                                                Technical Details:
                                            </Typography>
                                            <Typography variant="caption" paragraph sx={{ fontFamily: 'monospace', display: 'block' }}>
                                                {errorState.details}
                                            </Typography>
                                        </>
                                    )}

                                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom sx={{ mt: 2 }}>
                                        Suggested Fix:
                                    </Typography>
                                    <Alert severity="info">
                                        {errorState?.suggestion || "Please check the logs below for more information."}
                                    </Alert>
                                </Box>
                            </DialogContent>
                            <DialogActions>
                                <Button onClick={handleCloseErrorDialog} autoFocus>
                                    Close
                                </Button>
                            </DialogActions>
                        </Dialog>

                        {!isTraining ? (
                            <Button
                                variant="contained"
                                size="large"
                                onClick={handleTrain}
                                disabled={!file}
                                startIcon={<PlayArrowIcon />}
                                sx={{ minWidth: 200, py: 1.5, fontSize: '1.1rem' }}
                            >
                                Start Retraining
                            </Button>
                        ) : (
                            <Button
                                variant="contained"
                                color="error"
                                size="large"
                                onClick={handleStop}
                                disabled={stopping}
                                startIcon={<StopCircleIcon />}
                                sx={{ minWidth: 200, py: 1.5, fontSize: '1.1rem' }}
                            >
                                {stopping ? "Stopping..." : "Stop Training"}
                            </Button>
                        )}
                    </Paper>

                    {/* Console Output Section */}
                    <Box sx={{ mt: 4 }}>
                        <Button
                            onClick={() => setIsConsoleOpen(!isConsoleOpen)}
                            endIcon={isConsoleOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                            startIcon={<TerminalIcon />}
                            sx={{ color: 'text.secondary', mb: 1, textTransform: 'none' }}
                        >
                            {isConsoleOpen ? "Hide Process Console" : "Show Process Console"}
                        </Button>

                        <Collapse in={isConsoleOpen}>
                            <Paper
                                ref={logsContainerRef}
                                sx={{ p: 2, bgcolor: '#1e1e1e', color: '#00ff00', borderRadius: 2, fontFamily: 'monospace', height: 350, overflowY: 'auto', textAlign: 'left', border: '1px solid #333' }}
                            >
                                <Typography variant="overline" color="text.secondary" sx={{ color: '#888', display: 'block', mb: 1, borderBottom: '1px solid #333' }}>
                                    Backend Process Log {isTraining ? "(LIVE)" : ""}
                                </Typography>
                                <Box sx={{ whiteSpace: 'pre-wrap', fontSize: '0.85rem' }}>
                                    {trainingLogs.length > 0 ? (
                                        trainingLogs.map((log, index) => (
                                            <div key={index} style={{ marginBottom: '4px', borderBottom: '1px solid #111' }}>{log}</div>
                                        ))
                                    ) : (
                                        <span style={{ color: '#555' }}>Variable initialization... Waiting for process start...</span>
                                    )}
                                </Box>
                            </Paper>
                        </Collapse>
                    </Box>
                </Grid>
            </Grid>
        </Box >
    );
};

export default Retraining;
