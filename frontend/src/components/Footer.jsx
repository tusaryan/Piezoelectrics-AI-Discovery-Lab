import React from 'react';
import { Box, Container, Grid, Typography, Link, Stack, IconButton, Tooltip, Divider } from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import TwitterIcon from '@mui/icons-material/Twitter';
import EmailIcon from '@mui/icons-material/Email';
import { motion } from 'framer-motion';

const FooterLink = ({ children, href }) => (
    <Link
        href={href}
        color="text.secondary"
        underline="none"
        sx={{
            display: 'block',
            mb: 1,
            transition: 'color 0.2s',
            '&:hover': { color: 'common.white' },
            cursor: 'pointer'
        }}
    >
        {children}
    </Link>
);

const SocialButton = ({ icon, label, href, username }) => (
    <Tooltip
        title={
            <Box sx={{ textAlign: 'center', p: 1 }}>
                <Typography variant="body2" fontWeight="bold">@{username}</Typography>
                <Typography variant="caption">View Profile</Typography>
            </Box>
        }
        arrow
        placement="top"
    >
        <IconButton
            component={motion.a}
            href={href}
            target="_blank"
            whileHover={{ scale: 1.2, rotate: 5, backgroundColor: 'rgba(255,255,255,0.1)' }}
            whileTap={{ scale: 0.9 }}
            sx={{ color: 'text.secondary', '&:hover': { color: 'common.white' } }}
        >
            {icon}
        </IconButton>
    </Tooltip>
);

const Footer = () => {
    return (
        <Box sx={{ bgcolor: '#000000', color: 'text.secondary', py: 8, mt: 'auto' }}>
            <Container maxWidth="xl">
                <Grid container spacing={4} justifyContent="space-between">
                    {/* Brand & Developer Info */}
                    <Grid item xs={12} md={4}>
                        <Typography variant="h5" color="common.white" fontWeight="800" gutterBottom sx={{ letterSpacing: '-0.02em' }}>
                            PIEZO.AI
                        </Typography>
                        <Typography variant="body2" paragraph sx={{ maxWidth: 300, mb: 3 }}>
                            Accelerating materials discovery with state-of-the-art machine learning models.
                        </Typography>

                        <Box sx={{ mb: 2 }}>
                            <Typography variant="subtitle2" color="common.white" gutterBottom>
                                Developed by
                            </Typography>
                            <Typography variant="h6" color="primary.main" fontWeight="bold">
                                Aryan
                            </Typography>
                            <Typography variant="caption" display="block" gutterBottom>
                                Full Stack AI Engineer
                            </Typography>
                        </Box>

                        <Stack direction="row" spacing={1}>
                            <SocialButton
                                icon={<GitHubIcon />}
                                label="GitHub"
                                href="https://github.com/tusaryan"
                                username="tusaryan"
                            />
                            <SocialButton
                                icon={<LinkedInIcon />}
                                label="LinkedIn"
                                href="https://linkedin.com/in/tusaryan"
                                username="tusaryan"
                            />
                            <SocialButton
                                icon={<TwitterIcon />}
                                label="Twitter"
                                href="#"
                                username="tusaryan"
                            />
                            <SocialButton
                                icon={<EmailIcon />}
                                label="Email"
                                href="mailto:contact@example.com"
                                username="contact"
                            />
                        </Stack>
                    </Grid>

                    {/* Links Columns */}
                    <Grid item xs={6} md={2}>
                        <Typography variant="subtitle1" color="common.white" fontWeight="bold" gutterBottom>
                            Discover
                        </Typography>
                        <FooterLink href="#">Virtual Lab</FooterLink>
                        <FooterLink href="#">Instant Results</FooterLink>
                        <FooterLink href="#">Data Insights</FooterLink>
                        <FooterLink href="#">Case Studies</FooterLink>
                        <FooterLink href="#">Research</FooterLink>
                    </Grid>

                    <Grid item xs={6} md={2}>
                        <Typography variant="subtitle1" color="common.white" fontWeight="bold" gutterBottom>
                            Resources
                        </Typography>
                        <FooterLink href="#">Documentation</FooterLink>
                        <FooterLink href="#">API Reference</FooterLink>
                        <FooterLink href="#">Community</FooterLink>
                        <FooterLink href="#">Blog</FooterLink>
                        <FooterLink href="#">Support</FooterLink>
                    </Grid>

                    <Grid item xs={6} md={2}>
                        <Typography variant="subtitle1" color="common.white" fontWeight="bold" gutterBottom>
                            Company
                        </Typography>
                        <FooterLink href="#">About Us</FooterLink>
                        <FooterLink href="#">Careers</FooterLink>
                        <FooterLink href="#">Privacy Policy</FooterLink>
                        <FooterLink href="#">Terms of Use</FooterLink>
                        <FooterLink href="#">Contact</FooterLink>
                    </Grid>
                </Grid>

                <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)', my: 4 }} />

                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
                    <Typography variant="body2">
                        © {new Date().getFullYear()} Piezo.AI. All rights reserved.
                    </Typography>
                    <Typography variant="body2">
                        Designed with precision.
                    </Typography>
                </Box>
            </Container>
        </Box>
    );
};

export default Footer;
