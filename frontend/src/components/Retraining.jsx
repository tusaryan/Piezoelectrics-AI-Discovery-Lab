
import React, { useState } from 'react';
import {
    Box, Typography, Button, Paper, Grid, Alert, LinearProgress, Stack,
    FormControl, InputLabel, Select, MenuItem, Slider, Tooltip, IconButton,
    Card, CardContent, TextField
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AutoModeIcon from '@mui/icons-material/AutoMode';
import TuneIcon from '@mui/icons-material/Tune';
import InfoIcon from '@mui/icons-material/Info';
import axios from 'axios';

const MODEL_INFO = {
    "Random Forest": "An ensemble learning method that constructs multiple decision trees. Good for handling non-linear data and reducing overfitting.",
    "XGBoost": "Extreme Gradient Boosting. Highly efficient and flexible. Great for structured data and often achieves state-of-the-art results.",
    "LightGBM": "Light Gradient Boosting Machine. Faster training speed and lower memory usage. efficient for large datasets.",
    "Gradient Boosting": "Builds models sequentially, with each new model correcting errors of the previous ones.",
    "SVM (SVR)": "Support Vector Regression. Effective in high-dimensional spaces. Uses kernels to handle non-linear relationships."
};

const PARAM_INFO = {
    "n_estimators": "The number of trees in the forest or ensemble. More trees can improve performance but increase training time.",
    "learning_rate": "Shrinks the contribution of each tree. Lower values require more trees but can lead to better generalization.",
    "max_depth": "The maximum depth of a tree. Deeper trees can model more complex relations but may overfit.",
    "C": "Regularization parameter. Controls the trade-off between smooth decision boundary and classifying training points correctly.",
    "epsilon": "Epsilon in the epsilon-SVR model. Specifies the epsilon-tube within which no penalty is associated in the training loss function."
};

const Retraining = ({ isTraining, progress, statusMessage }) => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState(null);
    const [error, setError] = useState(null);
    const [mode, setMode] = useState('auto'); // 'auto' or 'manual'

    // Manual Config State
    const [selectedModel, setSelectedModel] = useState('Random Forest');
    const [nEstimators, setNEstimators] = useState(100);
    const [learningRate, setLearningRate] = useState(0.1);
    const [maxDepth, setMaxDepth] = useState(5);
    const [cParam, setCParam] = useState(100);
    const [epsilon, setEpsilon] = useState(0.1);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
        setMessage(null);
        setError(null);
    };

    const handleTrain = async () => {
        if (!file) {
            setError("Please upload a dataset CSV file.");
            return;
        }

        setMessage(null);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('mode', mode); // This is actually 'model_type' in backend if manual, but backend expects 'model_type' param separately if not auto.

        // Adjusting to backend expectation:
        // Backend expects: model_type="Auto" or specific name.
        if (mode === 'auto') {
            formData.append('model_type', 'Auto');
        } else {
            formData.append('model_type', selectedModel);
            formData.append('n_estimators', nEstimators);
            formData.append('learning_rate', learningRate);
            formData.append('max_depth', maxDepth);
            formData.append('c_param', cParam);
            formData.append('epsilon', epsilon);
        }

        try {
            await axios.post('http://localhost:8000/train', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
        } catch (err) {
            setError(err.response?.data?.detail || "Training failed.");
        }
    };

    const renderParamSlider = (label, value, setValue, min, max, step, paramKey) => (
        <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2" gutterBottom>{label}: {value}</Typography>
                <Tooltip title={PARAM_INFO[paramKey]} arrow placement="right">
                    <IconButton size="small" sx={{ ml: 0.5, p: 0.5 }}>
                        <InfoIcon fontSize="small" color="action" />
                    </IconButton>
                </Tooltip>
            </Box>
            <Slider
                value={value}
                onChange={(e, val) => setValue(val)}
                min={min}
                max={max}
                step={step}
                valueLabelDisplay="auto"
                disabled={isTraining}
            />
        </Box>
    );

    const renderManualConfig = () => (
        <Card variant="outlined" sx={{ mb: 4, textAlign: 'left', bgcolor: '#f8f9fa' }}>
            <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    Configuration
                    <Tooltip title="Manually select and tune the model. Comparison graphs will still be generated against other default models." arrow>
                        <IconButton size="small" sx={{ ml: 1 }}>
                            <InfoIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                </Typography>

                <FormControl fullWidth sx={{ mb: 3 }}>
                    <InputLabel>Select Model</InputLabel>
                    <Select
                        value={selectedModel}
                        label="Select Model"
                        onChange={(e) => setSelectedModel(e.target.value)}
                        disabled={isTraining}
                    >
                        {Object.keys(MODEL_INFO).map(model => (
                            <MenuItem key={model} value={model}>
                                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                                    {model}
                                    <Tooltip title={MODEL_INFO[model]} arrow placement="right">
                                        <IconButton size="small" sx={{ ml: 1, p: 0 }} onClick={(e) => e.stopPropagation()}>
                                            <InfoIcon fontSize="small" color="action" />
                                        </IconButton>
                                    </Tooltip>
                                </Box>
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>

                {['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting'].includes(selectedModel) && (
                    <>
                        {renderParamSlider("Number of Estimators", nEstimators, setNEstimators, 10, 1000, 10, "n_estimators")}
                        {renderParamSlider("Max Depth", maxDepth, setMaxDepth, 1, 20, 1, "max_depth")}
                    </>
                )}

                {['XGBoost', 'LightGBM', 'Gradient Boosting'].includes(selectedModel) && (
                    renderParamSlider("Learning Rate", learningRate, setLearningRate, 0.001, 1.0, 0.001, "learning_rate")
                )}

                {selectedModel === 'SVM (SVR)' && (
                    <>
                        {renderParamSlider("C (Regularization)", cParam, setCParam, 0.1, 1000, 0.1, "C")}
                        {renderParamSlider("Epsilon", epsilon, setEpsilon, 0.001, 1.0, 0.001, "epsilon")}
                    </>
                )}
            </CardContent>
        </Card>
    );

    return (
        <Box sx={{ width: '100%', maxWidth: 1000, mx: 'auto' }}>
            <Box sx={{ textAlign: 'center', mb: 6 }}>
                <Typography variant="h3" gutterBottom color="primary">Model Retraining</Typography>
                <Typography variant="body1" color="text.secondary">
                    Upload new data to improve model accuracy. Choose between intelligent auto-tuning or manual configuration.
                </Typography>
            </Box>

            <Grid container spacing={4} justifyContent="center">
                <Grid size={{ xs: 12, md: 8 }}>
                    <Paper sx={{ p: 4, borderRadius: 4, textAlign: 'center' }}>
                        <Stack direction="row" spacing={2} justifyContent="center" sx={{ mb: 4 }}>
                            <Button
                                variant={mode === 'auto' ? "contained" : "outlined"}
                                startIcon={<AutoModeIcon />}
                                onClick={() => setMode('auto')}
                                size="large"
                                disabled={isTraining}
                            >
                                Intelligent Auto-Tune
                            </Button>
                            <Button
                                variant={mode === 'manual' ? "contained" : "outlined"}
                                startIcon={<TuneIcon />}
                                onClick={() => setMode('manual')}
                                size="large"
                                disabled={isTraining}
                            >
                                Manual Configuration
                            </Button>
                        </Stack>

                        {mode === 'manual' && renderManualConfig()}

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

                        {isTraining && (
                            <Box sx={{ width: '100%', mb: 3 }}>
                                <LinearProgress variant="determinate" value={progress} />
                                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                    {statusMessage} ({progress}%)
                                </Typography>
                            </Box>
                        )}

                        {!isTraining && progress === 100 && (
                            <Alert severity="success" sx={{ mb: 3 }}>Training successfully completed! You can now use the new models.</Alert>
                        )}

                        {message && <Alert severity="info" sx={{ mb: 3 }}>{message}</Alert>}
                        {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

                        <Button
                            variant="contained"
                            size="large"
                            onClick={handleTrain}
                            disabled={isTraining || !file}
                            sx={{ minWidth: 200, py: 1.5, fontSize: '1.1rem' }}
                        >
                            {isTraining ? "Training in Progress..." : "Start Retraining"}
                        </Button>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default Retraining;
