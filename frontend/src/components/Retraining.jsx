
import React, { useState } from 'react';
import { Box, Typography, Button, Paper, Grid, Alert, LinearProgress, Stack } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AutoModeIcon from '@mui/icons-material/AutoMode';
import TuneIcon from '@mui/icons-material/Tune';
import axios from 'axios';

const Retraining = ({ isTraining, progress, statusMessage }) => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState(null);
    const [error, setError] = useState(null);
    const [mode, setMode] = useState('auto'); // 'auto' or 'manual'

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
        formData.append('mode', mode);

        try {
            const response = await axios.post('http://localhost:8000/train', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            // Don't set message here, wait for progress to complete or use statusMessage
        } catch (err) {
            setError(err.response?.data?.detail || "Training failed.");
        }
    };

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
