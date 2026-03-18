import Post from '../models/Post.js';
import Comment from '../models/Comment.js';
import User from '../models/User.js';
import Violation from '../models/Violation.js';
import ModerationLog from '../models/ModerationLog.js';
import { predictMessage } from '../services/mlClient.js';

const VIOLATION_THRESHOLD = 3;
const AUTO_SUSPEND_CREDIBILITY_THRESHOLD = 20;

const applyAutoSuspensionIfNeeded = async (user, sourceLabel) => {
  if (user.credibilityScore >= AUTO_SUSPEND_CREDIBILITY_THRESHOLD || user.isSuspended) {
    return;
  }

  user.isSuspended = true;
  user.suspensionReason = `Account blocked: credibility score dropped below ${AUTO_SUSPEND_CREDIBILITY_THRESHOLD} after repeated cyberbullying violations (${sourceLabel}).`;

  await ModerationLog.create({
    user: user._id,
    actionTaken: 'Account Suspended',
    reason: user.suspensionReason,
  });
};

// @desc    Create a post
// @route   POST /api/social/posts
// @access  Private
export const createPost = async (req, res) => {
  try {
    const { content } = req.body;
    
    if (!content) {
      return res.status(400).json({ error: 'Content is required' });
    }

    // Call ML API
    let isBullying = false;
    let toxicityScore = 0;
    let abusiveWords = [];

    try {
      const mlResponse = await predictMessage(content);
      isBullying = mlResponse.prediction === 1;
      toxicityScore = mlResponse.toxicityScore;
      abusiveWords = mlResponse.abusiveWords;
    } catch (mlError) {
      console.error('ML API Error during post creation:', mlError.message);
    }

    const user = await User.findById(req.user._id);

    if (isBullying) {
      user.credibilityScore = Math.max(0, user.credibilityScore - 10);
      await applyAutoSuspensionIfNeeded(user, 'posts');
      await user.save();

      const violation = await Violation.create({
        user: user._id,
        type: 'Cyberbullying',
        severity: Math.ceil(toxicityScore * 10),
        description: `Attempted to create a toxic post containing: ${abusiveWords.join(', ')}`,
      });

      const userViolationsCount = await Violation.countDocuments({ user: user._id });
          
      if (userViolationsCount >= VIOLATION_THRESHOLD) {
        await ModerationLog.create({
          user: user._id,
          actionTaken: 'Credibility Reduced',
          reason: `Exceeded violation threshold (${userViolationsCount} violations) from bad posts`,
        });
      }

      return res.status(403).json({ 
        error: 'Post blocked: Cyberbullying detected.',
        toxicityScore,
        abusiveWords
      });
    }

    // Safe Post
    user.credibilityScore = Math.min(100, user.credibilityScore + 1);
    await user.save();

    const post = await Post.create({
      author: req.user._id,
      content,
    });

    const populatedPost = await Post.findById(post._id).populate('author', 'username credibilityScore');
    res.status(201).json(populatedPost);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// @desc    Get all posts
// @route   GET /api/social/posts
// @access  Private
export const getPosts = async (req, res) => {
  try {
    const posts = await Post.find()
      .populate('author', 'username credibilityScore')
      .sort({ createdAt: -1 });
    res.json(posts);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// @desc    Add comment to a post
// @route   POST /api/social/posts/:id/comments
// @access  Private
export const addComment = async (req, res) => {
  try {
    const { content } = req.body;
    const postId = req.params.id;

    if (!content) {
      return res.status(400).json({ error: 'Content is required' });
    }

    const post = await Post.findById(postId);
    if (!post) {
      return res.status(404).json({ error: 'Post not found' });
    }

    // Call ML API
    let isBullying = false;
    let toxicityScore = 0;
    let abusiveWords = [];

    try {
      const mlResponse = await predictMessage(content);
      isBullying = mlResponse.prediction === 1;
      toxicityScore = mlResponse.toxicityScore;
      abusiveWords = mlResponse.abusiveWords;
    } catch (mlError) {
      console.error('ML API Error during comment creation:', mlError.message);
    }

    const user = await User.findById(req.user._id);

    if (isBullying) {
      user.credibilityScore = Math.max(0, user.credibilityScore - 10);
      await applyAutoSuspensionIfNeeded(user, 'comments');
      await user.save();

      const violation = await Violation.create({
        user: user._id,
        type: 'Cyberbullying',
        severity: Math.ceil(toxicityScore * 10),
        description: `Attempted to create a toxic comment containing: ${abusiveWords.join(', ')}`,
      });

      const userViolationsCount = await Violation.countDocuments({ user: user._id });
          
      if (userViolationsCount >= VIOLATION_THRESHOLD) {
        await ModerationLog.create({
          user: user._id,
          actionTaken: 'Credibility Reduced',
          reason: `Exceeded violation threshold (${userViolationsCount} violations) from bad comments`,
        });
      }

      return res.status(403).json({ 
        error: 'Comment blocked: Cyberbullying detected.',
        toxicityScore,
        abusiveWords
      });
    }

    // Safe Comment
    user.credibilityScore = Math.min(100, user.credibilityScore + 1);
    await user.save();

    const comment = await Comment.create({
      author: req.user._id,
      post: postId,
      content,
    });

    const populatedComment = await Comment.findById(comment._id).populate('author', 'username credibilityScore');
    res.status(201).json(populatedComment);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// @desc    Get comments for a post
// @route   GET /api/social/posts/:id/comments
// @access  Private
export const getComments = async (req, res) => {
  try {
    const postId = req.params.id;
    const comments = await Comment.find({ post: postId })
      .populate('author', 'username credibilityScore')
      .sort({ createdAt: 1 });
    res.json(comments);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
