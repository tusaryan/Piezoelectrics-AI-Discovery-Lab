
import React, { useState } from 'react';
import { Box, Typography, Button, Paper, Grid, Alert, LinearProgress, Stack } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AutoModeIcon from '@mui/icons-material/AutoMode';
import TuneIcon from '@mui/icons-material/Tune';
import axios from 'axios';

const Retraining = ({ onTrainingComplete }) => {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
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

        setLoading(true);
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
            setMessage(response.data.message);
            if (onTrainingComplete) {
                onTrainingComplete();
            }
        } catch (err) {
            setError(err.response?.data?.detail || "Training failed.");
        } finally {
            setLoading(false);
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
                            >
                                Intelligent Auto-Tune
                            </Button>
                            <Button
                                variant={mode === 'manual' ? "contained" : "outlined"}
                                startIcon={<TuneIcon />}
                                onClick={() => setMode('manual')}
                                size="large"
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
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                    bgcolor: 'action.hover',
                                    borderColor: 'primary.main',
                                    transform: 'scale(1.02)'
                                },
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                minHeight: 200
                            }}
                            component="label"
                        >
                            <input
                                type="file"
                                hidden
                                accept=".csv"
                                onChange={handleFileChange}
                            />
                            <CloudUploadIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                            <Typography variant="h6" color="text.primary" sx={{ wordBreak: 'break-word', px: 2 }}>
                                {file ? file.name : "Click to Upload Dataset (CSV)"}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Drag and drop or browse
                            </Typography>
                        </Box>

                        {loading && (
                            <Box sx={{ width: '100%', mb: 3 }}>
                                <LinearProgress />
                                <Typography variant="caption" color="text.secondary">Training in progress... This may take a few minutes.</Typography>
                            </Box>
                        )}

                        {message && <Alert severity="success" sx={{ mb: 3 }}>{message}</Alert>}
                        {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

                        <Button
                            variant="contained"
                            size="large"
                            onClick={handleTrain}
                            disabled={loading || !file}
                            sx={{ minWidth: 200, py: 1.5, fontSize: '1.1rem' }}
                        >
                            {loading ? "Training..." : "Start Retraining"}
                        </Button>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default Retraining;
