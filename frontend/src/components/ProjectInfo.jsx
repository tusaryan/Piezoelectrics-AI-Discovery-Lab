import React from 'react';
import { Box, Typography, Paper, Grid, Chip, Divider, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import SpeedIcon from '@mui/icons-material/Speed';
import NatureIcon from '@mui/icons-material/Nature';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import StorageIcon from '@mui/icons-material/Storage';
import BiotechIcon from '@mui/icons-material/Biotech';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const ProjectInfo = () => {
    return (
        <Box sx={{ width: '100%', maxWidth: 1200, mx: 'auto' }}>
            <Box sx={{ textAlign: 'center', mb: 8 }}>
                <Typography variant="h3" gutterBottom color="primary" sx={{ fontWeight: 'bold' }}>
                    About The Project
                </Typography>
                <Typography variant="h5" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto' }}>
                    AI-Assisted Discovery of New Lead-Free Piezoelectrics
                </Typography>
            </Box>

            <Grid container spacing={4}>
                {/* Problem & Solution */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 4, height: '100%', borderRadius: 4 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                            <NatureIcon color="error" sx={{ fontSize: 40, mr: 2 }} />
                            <Typography variant="h4">The Problem</Typography>
                        </Box>
                        <Typography variant="body1" paragraph>
                            For decades, <strong>Lead Zirconate Titanate (PZT)</strong> has been the industry standard for piezoelectric devices. However, it poses significant challenges:
                        </Typography>
                        <List>
                            <ListItem>
                                <ListItemIcon><CheckCircleIcon color="error" /></ListItemIcon>
                                <ListItemText primary="Toxicity: Contains over 60% lead by weight." />
                            </ListItem>
                            <ListItem>
                                <ListItemIcon><CheckCircleIcon color="error" /></ListItemIcon>
                                <ListItemText primary="Environmental Impact: E-waste contamination." />
                            </ListItem>
                            <ListItem>
                                <ListItemIcon><CheckCircleIcon color="error" /></ListItemIcon>
                                <ListItemText primary="Regulatory Pressure: Restrictions like RoHS." />
                            </ListItem>
                        </List>
                        <Typography variant="body1">
                            Finding a lead-free replacement is difficult due to the vast search space of chemical combinations.
                        </Typography>
                    </Paper>
                </Grid>

                <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 4, height: '100%', borderRadius: 4, bgcolor: 'primary.main', color: 'white' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                            <ScienceIcon sx={{ fontSize: 40, mr: 2, color: 'white' }} />
                            <Typography variant="h4" color="white">The Solution</Typography>
                        </Box>
                        <Typography variant="body1" paragraph sx={{ color: 'rgba(255,255,255,0.9)' }}>
                            We leverage <strong>Materials Informatics</strong> and <strong>Machine Learning</strong> to accelerate discovery.
                        </Typography>
                        <List sx={{ color: 'white' }}>
                            <ListItem>
                                <ListItemIcon><CheckCircleIcon sx={{ color: 'white' }} /></ListItemIcon>
                                <ListItemText primary="Data Collection: Aggregated data on KNN-based ceramics." />
                            </ListItem>
                            <ListItem>
                                <ListItemIcon><CheckCircleIcon sx={{ color: 'white' }} /></ListItemIcon>
                                <ListItemText primary="Feature Engineering: Smart chemical parser." />
                            </ListItem>
                            <ListItem>
                                <ListItemIcon><CheckCircleIcon sx={{ color: 'white' }} /></ListItemIcon>
                                <ListItemText primary="Predictive Modeling: Ensemble learning (XGBoost, LightGBM)." />
                            </ListItem>
                            <ListItem>
                                <ListItemIcon><CheckCircleIcon sx={{ color: 'white' }} /></ListItemIcon>
                                <ListItemText primary="Web Application: Interactive Virtual Laboratory." />
                            </ListItem>
                        </List>
                    </Paper>
                </Grid>

                {/* Key Features */}
                <Grid size={{ xs: 12 }}>
                    <Box sx={{ my: 6 }}>
                        <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', mb: 4 }}>
                            Key Features
                        </Typography>
                        <Grid container spacing={3}>
                            <Grid size={{ xs: 12, md: 4 }}>
                                <Paper sx={{ p: 3, textAlign: 'center', height: '100%' }}>
                                    <SpeedIcon color="primary" sx={{ fontSize: 50, mb: 2 }} />
                                    <Typography variant="h6" gutterBottom>Interactive Prediction</Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Instantly predict d33 and Tc for complex chemical formulas.
                                    </Typography>
                                </Paper>
                            </Grid>
                            <Grid size={{ xs: 12, md: 4 }}>
                                <Paper sx={{ p: 3, textAlign: 'center', height: '100%' }}>
                                    <AutoGraphIcon color="primary" sx={{ fontSize: 50, mb: 2 }} />
                                    <Typography variant="h6" gutterBottom>Automated Training</Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Upload new datasets and retrain models with auto-tuning.
                                    </Typography>
                                </Paper>
                            </Grid>
                            <Grid size={{ xs: 12, md: 4 }}>
                                <Paper sx={{ p: 3, textAlign: 'center', height: '100%' }}>
                                    <StorageIcon color="primary" sx={{ fontSize: 50, mb: 2 }} />
                                    <Typography variant="h6" gutterBottom>Model Management</Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        One-click deployment of best-performing models.
                                    </Typography>
                                </Paper>
                            </Grid>
                        </Grid>
                    </Box>
                </Grid>

                {/* Scientific Validation & Future Work */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 4, height: '100%', borderRadius: 4 }}>
                        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                            <BiotechIcon color="primary" sx={{ mr: 1 }} /> Scientific Validation
                        </Typography>
                        <Divider sx={{ mb: 2 }} />
                        <Typography variant="body1" paragraph>
                            Our models are validated using rigorous methodologies:
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                            <Chip label="Random Forest" variant="outlined" />
                            <Chip label="XGBoost" variant="outlined" />
                            <Chip label="LightGBM" variant="outlined" />
                            <Chip label="80/20 Split" color="primary" />
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                            We use R² and RMSE metrics to ensure high accuracy on unseen data, mirroring academic standards.
                        </Typography>
                    </Paper>
                </Grid>

                <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 4, height: '100%', borderRadius: 4 }}>
                        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                            <AutoGraphIcon color="secondary" sx={{ mr: 1 }} /> Future Roadmap
                        </Typography>
                        <Divider sx={{ mb: 2 }} />
                        <List dense>
                            <ListItem>
                                <ListItemText
                                    primary="Inverse Design"
                                    secondary="Using Genetic Algorithms to generate new compositions."
                                />
                            </ListItem>
                            <ListItem>
                                <ListItemText
                                    primary="Deep Learning"
                                    secondary="Graph Neural Networks (GNNs) for crystal structures."
                                />
                            </ListItem>
                            <ListItem>
                                <ListItemText
                                    primary="Microstructure Analysis"
                                    secondary="Computer Vision for SEM image analysis."
                                />
                            </ListItem>
                            <ListItem>
                                <ListItemText
                                    primary="Advanced Chemical Formula Parser"
                                    secondary="Improve the parser to handle complex chemical formulas."
                                />
                            </ListItem>
                        </List>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default ProjectInfo;
