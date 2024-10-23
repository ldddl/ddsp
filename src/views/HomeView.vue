<script>
// import NavBar from '../components/NavBar.vue'

export default {
  components: {
  }
}
</script>

<template>
  <div class="container mt-4">
    <router-link to="/inference" class="button-style p-2 text-decoration-none">推理</router-link>
    <router-link to="/training" class="button-style p-2 text-decoration-none">訓練</router-link>
    <router-link to="/audio-slicing" class="button-style p-2 text-decoration-none">智能音頻切片</router-link>
  </div>
  <div class="container pt-2">
    <router-view class="content-style"></router-view>
  </div>
</template>

<style lang="scss" scoped>
  // .button-style:focus {
  //   outline: 2px solid black;
  //   // border-radius: 10px 10px;
  // }
</style>